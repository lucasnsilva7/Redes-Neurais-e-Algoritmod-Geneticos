import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

### Definindo classes

class MLP(nn.Module):
    def __init__(self, num_dados_entrada, neuronios_c1, neuronios_c2, num_targets):
        super().__init__()
        
        self.camadas = nn.Sequential(
            nn.Linear(num_dados_entrada, neuronios_c1),
            nn.Sigmoid(),
            nn.Linear(neuronios_c1, neuronios_c2),
            nn.Sigmoid(),
            nn.Linear(neuronios_c2, num_targets),
        )
        
    def forward(self, x):
        x = self.camadas(x)
        return x
    
### Definindo o dataset

df = sns.load_dataset("iris")

### Dividindo os dados em treino e teste

TAMANHO_TESTE = 0.1

indices = df.index
indices_treino, indices_teste = train_test_split(
    indices, test_size=TAMANHO_TESTE
)

df_treino = df.loc[indices_treino]
df_teste = df.loc[indices_teste]

# P/ treino da rede
x_treino = df_treino["petal_length"].values
y_verdadeiro_treino = df_treino["petal_width"].values

x_treino = torch.tensor(x_treino)
x_treino = x_treino.view(-1, 1)
y_verdadeiro_treino = torch.tensor(y_verdadeiro_treino)
y_verdadeiro_treino = y_verdadeiro_treino.view(-1, 1)

# P/ validação da rede
x_validacao = df_teste["petal_length"].values
y_verdadeiro_validacao = df_teste["petal_width"].values

x_validacao = torch.tensor(x_validacao)
x_validacao = x_validacao.view(-1, 1)
y_verdadeiro_validacao = torch.tensor(y_verdadeiro_validacao)
y_verdadeiro_validacao = y_verdadeiro_validacao.view(-1,1)
x_treino = x_treino.float()
y_verdadeiro_treino = y_verdadeiro_treino.float()

x_validacao = x_validacao.float()
y_verdadeiro_validacao = y_verdadeiro_validacao.float()

NUM_DADOS_DE_ENTRADA = 1
NUM_DADOS_DE_SAIDA = 1
NEURONIOS_C1 = 3
NEURONIOS_C2 = 2

minha_mlp = MLP(
    NUM_DADOS_DE_ENTRADA, NEURONIOS_C1, NEURONIOS_C2, NUM_DADOS_DE_SAIDA
)
y_prev = minha_mlp(x_treino)

TAXA_DE_APRENDIZADO = 0.005

otimizador = optim.SGD(minha_mlp.parameters(), lr=TAXA_DE_APRENDIZADO)
fn_perda = nn.MSELoss()

NUMERO_EPOCAS_SEM_MELHORA = 12
MAXIMO_EPOCAS = 100000

melhor_rmse = float('inf')
cont = 0
cont_maximo = 0
epoca_melhor_modelo = 0

inicio = time.time()

while cont < NUMERO_EPOCAS_SEM_MELHORA and cont_maximo < MAXIMO_EPOCAS:
    
    # Treino
    minha_mlp.train()

    y_pred = minha_mlp(x_treino)
    otimizador.zero_grad()
    loss = fn_perda(y_verdadeiro_treino, y_pred)
    loss.backward()
    otimizador.step()

    # Teste
    minha_mlp.eval()

    with torch.no_grad():
        y_pred = minha_mlp(x_validacao)
        
    RMSE = mean_squared_error(y_verdadeiro_validacao, y_pred, squared=False)
    
    if RMSE < melhor_rmse:
        melhor_rmse = RMSE
        cont = 0
        epoca_melhor_modelo = cont_maximo
        melhor_desempenho = torch.save(minha_mlp.state_dict(), 'checkpoint.pt')
        
    else:
        cont += 1

    cont_maximo += 1
    print(f"Época {cont_maximo} | RMSE: {RMSE:.4f}")
    
fim = time.time()
tempo = fim - inicio

print()
print(f"Tempo rodar o código: {tempo:.2f} segundos")

minha_mlp.load_state_dict(torch.load('checkpoint.pt'))
print(f"O melhor modelo foi obtido na época {epoca_melhor_modelo}, com RMSE de {melhor_rmse:.4f}")