from src.modules.jaccard import JaccardModule
from src.modules.levenshtein import LevenshteinModule
from src.modules.ner import NERModule
from src.modules.lstm import LSTMModule

texts = [
    'Путин встретился с Трампом в Москве для обсуждения международных вопросов',
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

print('Testing Ner')
ner = NERModule()
logits = ner.get_logits(texts)
print(f'logits shape: {logits.shape}')
print(f'Logits:\n{logits}')
print(f'Entities:')
for i, text in enumerate(texts):
    ents = ner._extract_entities(text)
    print(f'  Text: {i} {ents}')
preds = ner.predict(texts)
print(f'Predictions (threshold={ner.default_threshold}):\n{preds}')
print()


print('Testing LSTMModule...')
train_pairs = [
    ('Путин встретился с Трампом', 'Президент России провел переговоры с американским лидером'),
    ('Погода сегодня солнечная', 'На улице тепло и ясно'),
    ('Путин в Москве', 'Погода в Питере'),
    ('Курс доллара вырос', 'Рубль укрепился к евро'),
]
train_labels = [1, 1, 0, 0]

lstm = LSTMModule(hidden_size=64, embed_dim=64, max_length=128, epochs=3, batch_size=2)
lstm.fit(train_pairs, train_labels)
logits = lstm.get_logits(texts)
print(f'Logits shape: {logits.shape}')
print(f'Logits:\n{logits}')
preds = lstm.predict(texts)
print(f'Predictions (threshold={lstm.default_threshold}):\n{preds}')
print()

print('Ok!')
