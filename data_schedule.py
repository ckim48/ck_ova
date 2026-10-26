import numpy as np


def dirichlet_partition(labels, users, alpha, seed, minimum_size=8):
    if alpha <= 0 or users < 1 or users * minimum_size > len(labels):
        raise ValueError("Invalid partition configuration.")
    rng = np.random.default_rng(seed)
    for attempt in range(2000):
        clients = [[] for _ in range(users)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(users, alpha)))
            offset = 0
            for user, count in enumerate(counts):
                clients[user].extend(indices[offset:offset + count].tolist())
                offset += count
        if min(map(len, clients)) >= minimum_size:
            return [rng.permutation(client) for client in clients]
    raise RuntimeError("Could not construct partition with minimum client size.")


def make_schedule(partitions, rounds=100, initial_rounds=50, requests=(51, 61, 71, 81, 91),
                  initial_fraction=0.8, deletion_fraction=0.5, requesting_users=2,
                  batch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    used = [set() for _ in partitions]
    deleted = [set() for _ in partitions]
    schedule = []
    for round_number in range(1, rounds + 1):
        visible = []
        for partition in partitions:
            initial_count = int(len(partition) * initial_fraction)
            if round_number <= initial_rounds:
                count = initial_count
            else:
                count = initial_count + (len(partition) - initial_count) * (
                    round_number - initial_rounds
                ) // (rounds - initial_rounds)
            visible.append(set(map(int, partition[:count])))
        forget = [[] for _ in partitions]
        if round_number in requests:
            selected = rng.choice(len(partitions), requesting_users, replace=False)
            for user in selected:
                candidates = sorted(used[user] - deleted[user])
                count = int(len(candidates) * deletion_fraction)
                forget[user] = rng.choice(candidates, count, replace=False).tolist()
                deleted[user].update(forget[user])
        retained = [sorted(visible[user] - deleted[user]) for user in range(len(partitions))]
        batches = []
        batch_seeds = []
        for user, indices in enumerate(retained):
            batch_seed = int(rng.integers(2 ** 31))
            batch_seeds.append(batch_seed)
            batch = np.random.default_rng(batch_seed).choice(
                indices, min(batch_size, len(indices)), replace=False
            ).tolist()
            used[user].update(batch)
            batches.append(batch)
        schedule.append({
            "round": round_number, "retained": retained, "forget": forget, "batches": batches,
            "batch_seeds": batch_seeds,
            "deleted": sorted(set().union(*deleted)),
        })
    return schedule
