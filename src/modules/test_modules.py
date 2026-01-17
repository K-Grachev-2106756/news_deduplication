from src.modules.jaccard import JaccardModule
from src.modules.levenshtein import LevenshteinModule

texts = [
    'Путин встретился с Байденом в Москве для обсуждения международных вопросов',
    'Президент России провел переговоры с американским лидером в столице',
    'Погода в Санкт-Петербурге сегодня солнечная и теплая'
]

print('Testing JaccardModule...')
jac = JaccardModule()
logits = jac.get_logits(texts)
print(f'Logits shape: {logits.shape}')
print(f'Logits:\n{logits}')
preds = jac.predict(texts)
print(f'Predictions (threshold={jac.default_threshold}):\n{preds}')
print()

print('Testing LevenshteinModule...')
lev = LevenshteinModule()
logits = lev.get_logits(texts)
print(f'Logits shape: {logits.shape}')
print(f'Logits:\n{logits}')
preds = lev.predict(texts)
print(f'Predictions (threshold={lev.default_threshold}):\n{preds}')
print()

print('Ok!')
