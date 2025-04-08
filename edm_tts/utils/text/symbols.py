SYMBOLS = sorted(list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? ')))
# SYMBOLS = list(set('abcdefghijklmnopqrstuvwxyz!\'(),-.:;? '))

SYMBOL_TO_ID = {symbol: i for i, symbol in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: symbol for i, symbol in enumerate(SYMBOLS)}
