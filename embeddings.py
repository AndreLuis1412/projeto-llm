
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

modelo = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

frases = [
    'Esqueci minha senha e não consigo entrar',
    'Não lembro a senha da minha conta',
    'Como faço para preparar um bolo?',
]

vetores = modelo.encode(frases)

similaridade_1 = cosine_similarity(
    [vetores[0]],
    [vetores[1]]
)[0][0]

similaridade_2 = cosine_similarity(
    [vetores[0]],
    [vetores[2]]
)[0][0]

print(f'Frase 1: {frases[0]}')
print(f'Frase 2: {frases[1]}')
print(f'Similaridade: {similaridade_1:.4f}')

print()

print(f'Frase 1: {frases[0]}')
print(f'Frase 3: {frases[2]}')
print(f'Similaridade: {similaridade_2:.4f}')
