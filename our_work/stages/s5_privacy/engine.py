"""S5a - wrap a client's model/optimizer/loader with Opacus DP-SGD.

Called ONCE per client (see s4_federated.client).  The accountant lives inside the returned
PrivacyEngine, so epsilon keeps accumulating over rounds; re-wrapping every round would reset it.
"""


def wrap_private(module, optimizer, loader, dp, total_epochs):
    """dp = dict(noise, clip, delta, target_eps).  If `target_eps` is set Opacus picks the noise
    multiplier to hit that epsilon after `total_epochs` epochs; otherwise `noise` is used as given.
    Returns (privacy_engine, wrapped_module, wrapped_optimizer, wrapped_loader)."""
    from opacus import PrivacyEngine
    pe = PrivacyEngine(secure_mode=False)
    if dp.get("target_eps"):
        net, opt, loader = pe.make_private_with_epsilon(
            module=module, optimizer=optimizer, data_loader=loader, target_epsilon=dp["target_eps"],
            target_delta=dp["delta"], epochs=total_epochs, max_grad_norm=dp["clip"])
    else:
        net, opt, loader = pe.make_private(
            module=module, optimizer=optimizer, data_loader=loader,
            noise_multiplier=dp["noise"], max_grad_norm=dp["clip"])
    return pe, net, opt, loader
