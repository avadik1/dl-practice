import torch


def adam_step(p, g, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
              weight_decay=0.0, mode="adam", bias_correction=True):
    """
    Один шаг Adam / AdamW для тензора p.

    p, g, m, v — тензоры одной формы (веса, градиент, 1-й и 2-й моменты).
    t — номер шага, начинается с 1.
    mode="adam"  — weight_decay работает как L2 в loss (так делает torch.optim.Adam)
    mode="adamw" — decoupled weight decay (так делает torch.optim.AdamW)

    Возвращает (p_new, m_new, v_new). Входные тензоры на месте не меняем.
    """
    # 1. L2-регуляризация: в loss добавлено слагаемое (wd/2) * ||p||^2.
    #    Чему равен его градиент? Прибавь его к g.
    if mode == "adam" and weight_decay != 0:
        g = ...  # TODO

    # 2. Decoupled weight decay: сжимаем веса напрямую, градиент НЕ трогаем.
    #    Точную формулу сверь с псевдокодом в документации torch.optim.AdamW.
    if mode == "adamw" and weight_decay != 0:
        p = ...  # TODO

    # 3. EMA первого момента (по градиентам)
    m = ...  # TODO

    # 4. EMA второго момента (по квадратам градиентов)
    v = ...  # TODO

    # 5. Bias correction: поправка на то, что m и v стартовали с нуля
    if bias_correction:
        m_hat = ...  # TODO
        v_hat = ...  # TODO
    else:
        m_hat, v_hat = m, v

    # 6. Обновление весов
    p = ...  # TODO

    return p, m, v
