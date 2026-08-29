"""
Tiers como constantes: pedir por intención sin memorizar strings.

No fijan la lista de tiers del gateway —esa vive en su `config/tiers.yaml` y se
descubre con `gateway.capabilities()`— sino que dan los tres estables un nombre
en el cliente, para que un typo sea un error de import y no un 502.

    from proxima_llm import SMART, SyncGateway
    gw.chat("...", model=SMART)
"""

# Los estables. El gateway puede tener más; `capabilities()` los lista todos.
SMART = "smart"  # el más capaz
CHEAP = "cheap"  # el más barato (por costo medido)
FAST = "fast"  # el de menor latencia (medida)
