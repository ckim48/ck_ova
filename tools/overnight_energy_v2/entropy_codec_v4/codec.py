"""Lossless packet coding of the ORIGINAL quantized unlearning integers."""
import struct
import zlib

import numpy as np


HEADER = struct.Struct('<4sBBHI')
MAGIC = b'ECV4'


def pack_codes(codes, bits):
    codes = np.asarray(codes,dtype=np.uint16).reshape(-1)
    bit_array = ((codes[:,None] >> np.arange(bits,dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(bit_array.reshape(-1),bitorder='little').tobytes()


def unpack_codes(data, count, bits):
    if len(data) != (count*bits+7)//8:
        raise ValueError('Incorrect packed integer length.')
    bit_array = np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='little')[:count*bits].reshape(count,bits)
    return (bit_array.astype(np.uint32)*(1<<np.arange(bits,dtype=np.uint32))).sum(axis=1).astype(np.uint16)


def encode(updates, bits):
    updates = np.asarray(updates,dtype=np.float64)
    if updates.ndim != 2 or not 2 <= bits <= 16 or len(updates) >= 65536:
        raise ValueError('Expected blocks-by-dimension updates and 2--16 bits.')
    if not np.any(updates):
        return b''
    levels = 2**(bits-1)-1
    maxima = np.max(np.abs(updates),axis=1)
    scale = (maxima/levels).astype(np.float32)
    safe = np.where(scale>0,scale.astype(np.float64),1.)
    signed = np.clip(np.rint(updates/safe[:,None]),-levels,levels).astype(np.int32)
    codes = (signed+levels).astype(np.uint16).reshape(-1)
    integer_bytes = codes.astype('<u1' if bits<=8 else '<u2').tobytes()
    raw = integer_bytes if bits in (8,16) else pack_codes(codes,bits)
    choices = [raw,zlib.compress(raw,9)]
    if raw!=integer_bytes:
        choices.append(zlib.compress(integer_bytes,9))
    codec = min(range(len(choices)),key=lambda i:len(choices[i]))
    data = choices[codec]
    return HEADER.pack(MAGIC,codec,bits,len(updates),len(data))+scale.astype('<f4').tobytes()+data


def decode(packet, blocks, dimension):
    if not packet:
        return np.zeros((blocks,dimension))
    if len(packet)<HEADER.size:
        raise ValueError('Truncated codec header.')
    magic,codec,bits,encoded_blocks,size = HEADER.unpack(packet[:HEADER.size])
    if magic!=MAGIC or codec not in (0,1,2) or not 2<=bits<=16 or encoded_blocks!=blocks:
        raise ValueError('Unsupported or inconsistent codec header.')
    start = HEADER.size+4*blocks
    if len(packet)!=start+size:
        raise ValueError('Incorrect compressed packet length.')
    scale = np.frombuffer(packet[HEADER.size:start],dtype='<f4').astype(np.float64)
    if not np.all(np.isfinite(scale)) or np.any(scale<0):
        raise ValueError('Invalid quantizer scales.')
    data = packet[start:]
    count = blocks*dimension
    if codec==2:
        raw = zlib.decompress(data)
        width = 1 if bits<=8 else 2
        if len(raw)!=count*width:
            raise ValueError('Incorrect decompressed integer length.')
        codes = np.frombuffer(raw,dtype='<u1' if width==1 else '<u2').astype(np.uint16)
    else:
        codes = unpack_codes(zlib.decompress(data) if codec==1 else data,count,bits)
    levels = 2**(bits-1)-1
    if np.any(codes>2*levels):
        raise ValueError('Invalid quantized integer code.')
    return (codes.astype(np.int32).reshape(blocks,dimension)-levels)*scale[:,None]


def fragment_bits(packet, blocks):
    """Known-order byte fragments; compressed length is signaled before design."""
    quotient,remainder = divmod(len(packet),blocks)
    lengths = np.full(blocks,quotient,dtype=np.int64)
    lengths[:remainder] += 1
    assert int(lengths.sum()) == len(packet)
    return lengths*8
