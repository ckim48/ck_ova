import torch


def cross_modal_update(gradient, cross_hessian, visual_dimension, text_dimension,
                       requested_kappa=1.0, xi=0.1):
    if not 0 < xi <= 1 or requested_kappa < 0:
        raise ValueError("Require 0 < xi <= 1 and nonnegative requested_kappa.")
    if cross_hessian.shape != (visual_dimension, text_dimension):
        raise ValueError("Cross-Hessian dimensions do not match parameter blocks.")
    norm = torch.linalg.matrix_norm(cross_hessian, ord=2)
    kappa = min(requested_kappa, (1 - xi) / max(float(norm), 1e-30))
    visual = gradient[:visual_dimension]
    text_end = visual_dimension + text_dimension
    text = gradient[visual_dimension:text_end]
    output = gradient.clone()
    output[:visual_dimension] = visual + kappa * cross_hessian @ text
    output[visual_dimension:text_end] = text + kappa * cross_hessian.T @ visual
    return output, kappa, float(norm)


def coupled_direction(loss_function, theta, forget_gradient, visual_dimension,
                      text_dimension, requested_kappa=1.0, xi=0.1):
    hessian = torch.autograd.functional.hessian(loss_function, theta)
    cross = hessian[:visual_dimension,
                    visual_dimension:visual_dimension + text_dimension]
    return cross_modal_update(forget_gradient, cross, visual_dimension,
                              text_dimension, requested_kappa, xi)
