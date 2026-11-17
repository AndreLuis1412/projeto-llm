
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Carregar o modelo de embeddings
modelo = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Criar nossa documentação
documentos = [
    'Para recuperar sua senha, acesse a tela de login, clique em Esqueci minha senha e informe seu e-mail cadastrado.',
    'As automações podem ser agendadas para execução diária, semanal ou mensal. O usuário pode configurar o horário de execução.',
    'Quando uma automação falha, consulte o histórico de execuções para verificar os erros registrados e identificar a causa.',
    'Para conectar uma conta à plataforma, acesse as configurações, selecione Integrações e informe as credenciais solicitadas.',
    'Os relatórios apresentam informações sobre as execuções das automações, incluindo quantidade de sucessos, falhas e tempo de processamento.'
]

# 3. Gerar os embeddings
vetores = modelo.encode(documentos)

# 4. Criar o banco vetorial
cliente = chromadb.Client()

colecao = cliente.create_collection(name='documentacao')

# 5. Armazenar documentos e embeddings
colecao.add(
    ids=[str(i) for i in range(len(documentos))],
    documents=documentos,
    embeddings=vetores.tolist()
)

print('Documentos armazenados com sucesso!')
print(f'Total armazenado: {colecao.count()}')

# 6. Fazer uma busca semântica
pergunta = 'Não consigo acessar minha conta, como faço para entrar novamente?'

# Transformar a pergunta em embedding
vetor_pergunta = modelo.encode(pergunta)

# Pesquisar os documentos mais semelhantes
resultados = colecao.query(
    query_embeddings=[vetor_pergunta.tolist()],
    n_results=3
)

# Exibir os resultados
print('\nPergunta:', pergunta)
print('\nDocumentos encontrados:')

for documento in resultados['documents'][0]:
    print('-', documento)

# Exibir os resultados e suas distâncias
print('\nPergunta:', pergunta)
print('\nDocumentos encontrados:')

for documento, distancia in zip(
    resultados['documents'][0],
    resultados['distances'][0]
):
    print(f'\nDocumento: {documento}')
    print(f'Distância: {distancia:.4f}')