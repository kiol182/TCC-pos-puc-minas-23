#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Comando para obter a versão Python que está sendo utilizada
get_ipython().system('python -V')


# In[2]:


# Código para importar a biblioteca pandas
import pandas as pd


# In[3]:


# Leitura do arquivo CSV com cotações de bitcoin e cria um DataFrame
dft_btc = pd.read_csv('btc_usd_historico.csv')

# Exibe o conteúdo do DataFrame
print(dft_btc)


# In[4]:


dft_btc.info()


# In[5]:


dft_btc.describe()


# In[6]:


dft_btc.isnull().sum()


# In[7]:


# Metodos para converter a string para tipo float dentro do df
def RetornarFloatParam(lista, param):
    nova_lista = [float(elemento.replace(param, "").replace(",", ".")) for elemento in lista]
    return nova_lista

def RetornarFloat(lista):
    return RetornarFloatParam(lista,".")

def RetornarPercentualFloat(lista):
    return RetornarFloatParam(lista,"%")

def RetornarVolFloat(lista):
    return  RetornarFloatParam(lista,"K")

# Metodo auxiliar para identificação do parametro
def Multiplicador(case):
    if case=='K': 
        return 1000
    if case=='M':
        return 1000000
    return 1

def TratarVolume(lista):    
    nova_lista = [float(elemento[:-1].replace(",", "."))*Multiplicador(elemento[-1]) for elemento in lista]
    return nova_lista


# In[8]:


dft_btc['Último']= RetornarFloat(dft_btc['Último'])
dft_btc['Abertura']= RetornarFloat(dft_btc['Abertura'])
dft_btc['Máxima']= RetornarFloat(dft_btc['Máxima'])
dft_btc['Mínima']= RetornarFloat(dft_btc['Mínima'])
dft_btc['Var%']= RetornarPercentualFloat(dft_btc['Var%'])
dft_btc['Vol.']= TratarVolume(dft_btc['Vol.'])

# Converter coluna data para o tipo datetime
dft_btc['Data'] = pd.to_datetime(dft_btc['Data'], format='%d.%m.%Y')


# In[9]:


dft_btc.info()


# In[10]:


# Leitura do arquivo CSV com cotações de ethereum e cria um DataFrame
dtf_Eth = pd.read_csv('eth_usd_historico.csv')

# Exibe o conteúdo do DataFrame
print(dtf_Eth)


# In[11]:


dtf_Eth.info()


# In[12]:


dtf_Eth.describe()


# In[13]:


dtf_Eth.isnull().sum()


# In[14]:


dtf_Eth['Último']= RetornarFloat(dtf_Eth['Último'])
dtf_Eth['Abertura']= RetornarFloat(dtf_Eth['Abertura'])
dtf_Eth['Máxima']= RetornarFloat(dtf_Eth['Máxima'])
dtf_Eth['Mínima']= RetornarFloat(dtf_Eth['Mínima'])
dtf_Eth['Var%']= RetornarPercentualFloat(dtf_Eth['Var%'])
dtf_Eth['Vol.']= TratarVolume(dtf_Eth['Vol.'])

# Converter coluna data para o tipo datetime
dtf_Eth['Data'] = pd.to_datetime(dtf_Eth['Data'], format='%d.%m.%Y')


# In[15]:


dtf_Eth.info()


# In[16]:


# Leitura do arquivo CSV com cotações do dolár e cria um DataFrame
dtf_usd = pd.read_csv('usd_brl_historico.csv')

# Exibe o conteúdo do DataFrame
print(dtf_usd)


# In[17]:


dtf_usd.info()
dtf_usd.isnull().sum()


# In[18]:


import matplotlib.pyplot as plt

dfplt = dtf_usd.head(25)

# Criar o gráfico de linha com marcadores
plt.plot(dfplt['Date'], dfplt['High'], marker='o')

# Adicionar rótulos e título
plt.xlabel('Data')
plt.ylabel('Cotação do Dólar')
plt.title('Cotação do Dólar nos 25 primeiros dias')

# Girar os rótulos do eixo x para facilitar a leitura
plt.xticks(rotation=45)

# Exibir o gráfico
plt.show()


# In[19]:


# Para evitar uso de registro duplicado, criamos um dataframe auxiliar
df = dtf_usd.head(900)

# Remover os campos que não serão utilizados
df = df.drop('Volume', axis=1)
df = df.drop('Adj Close', axis=1)

# Converte a coluna 'Date' para o tipo datetime
df['Date'] = pd.to_datetime(df['Date'])

# Define a coluna 'Date' como o índice do dataframe
df = df.set_index('Date')

# Cria um novo dataframe com todos os dias entre a primeira e a última data
data_inicio = df.index.min()
data_fim = df.index.max()
df_completo = pd.DataFrame(index=pd.date_range(data_inicio, data_fim), columns=['Open','High','Low','Close'])

# Preenche o novo dataframe com as cotações disponíveis no dataframe original
df_completo.update(df)

# Preenche os dias ausentes com a última cotação disponível
df_completo['Close'] = df_completo['Close'].ffill()
df_completo['Open'] = df_completo['Open'].ffill()
df_completo['High'] = df_completo['High'].ffill()
df_completo['Low'] = df_completo['Low'].ffill()

# Reverte a coluna 'Date' de volta para uma coluna do dataframe
dtf_usd = df_completo.reset_index()
dtf_usd['Data']= dtf_usd['index']
dtf_usd = dtf_usd.drop('index', axis=1)

# Renomear colunas para melhor identificação
dtf_usd = dtf_usd.rename(columns={'Close': 'Dl_Fechamento'
                                  ,'Open': 'Dl_Abertura'
                                  ,'High': 'Dl_Maior'
                                  ,'Low': 'Dl_Menor'})   


# In[20]:


# Renomear colunas para melhor identificação
dft_btc = dft_btc.rename(columns={'Último': 'Bt_Fechamento',
                                  'Abertura': 'Bt_Abertura',
                                  'Máxima': 'Bt_Maior',
                                  'Mínima': 'Bt_Menor',
                                  'Vol.': 'Bt_Volume',
                                  'Var%': 'Bt_Variacao'})


# In[21]:


# Renomear colunas para melhor identificação
dtf_Eth = dtf_Eth.rename(columns={'Último': 'Et_Fechamento',
                                  'Abertura': 'Et_Abertura',
                                  'Máxima': 'Et_Maior',
                                  'Mínima': 'Et_Menor',
                                  'Vol.': 'Et_Volume',
                                  'Var%': 'Et_Variacao'})


# In[22]:


# Colocar campo data com index para unir dataframes
dft_btc = dft_btc.set_index('Data')
dtf_usd = dtf_usd.set_index('Data')

# junção dataframes bitcoin e dolar
dt_bt_dl=dtf_usd.join(dft_btc)

dt_bt_dl.info()


# In[23]:


dt_bt_dl.describe()


# In[24]:


dt_bt_dl.isnull().sum()


# In[25]:


# Carregar o dataframe com as cotações de Dólar
df = dt_bt_dl

plt.boxplot([df['Dl_Abertura'], df['Dl_Maior'], df['Dl_Menor'], df['Dl_Fechamento']])

# Definir os rótulos dos eixos x e y
plt.xticks([1,2,3,4], ['Dl_Abertura','Dl_Maior','Dl_Menor','Dl_Fechamento'])
plt.ylabel('Valor')

# Definir um título para o gráfico
plt.title('Boxplot dos Valores do Dólar')

# Exibir o gráfico
plt.show()


# In[26]:


# Carregar o dataframe com as cotações de criptomoedas
df = dt_bt_dl

# Definir o tamanho do intervalo de média (por exemplo, 7 dias)
intervalo = '1M'

# Calcular a média dos preços em intervalos de tempo
df_media = df.resample(intervalo).mean()
df_media_outro_campo = df.resample(intervalo).mean()  

# Configurar o tamanho da figura
plt.figure(figsize=(10, 6))

# Plotar o gráfico de linhas da média dos preços
plt.plot(df_media.index, df_media['Dl_Fechamento'], marker='o', linestyle='-', color='b', label='Fechamento')
plt.plot(df_media.index, df_media_outro_campo['Dl_Abertura'], linestyle='--', color='r', label='Abertura')
plt.plot(df_media.index, df_media['Dl_Maior'], marker='o', linestyle='-', color='y', label='Maior')
plt.plot(df_media.index, df_media_outro_campo['Dl_Menor'], linestyle='--', color='g', label='Menor')

# Configurar os rótulos dos eixos
plt.xlabel('Data')
plt.ylabel('Média de Preço')

# Configurar o título do gráfico
plt.title('Média de Preço do dolar ao Longo do Periodo')
plt.legend(loc='lower right')
# Rotacionar os rótulos do eixo x para facilitar a leitura
plt.xticks(rotation=45)

# Exibir o gráfico
plt.show()


# In[27]:


# Armazenando o dataframe completo (desde 01/01/2020)
dt_bt_dl_completo = dt_bt_dl

# Filtrando os dados com inicio em 11-03-2020
dt_bt_dl = dt_bt_dl.loc['2020-03-11':]


# In[28]:


# Carregar o dataframe com as cotações
df = dt_bt_dl

plt.boxplot([df['Dl_Abertura'], df['Dl_Maior'], df['Dl_Menor'], df['Dl_Fechamento']])

# Definir os rótulos dos eixos x e y
plt.xticks([1,2,3,4], ['Dl_Abertura','Dl_Maior','Dl_Menor','Dl_Fechamento'])
plt.ylabel('Valor')

# Definir um título para o gráfico
plt.title('Boxplot dos Valores do Dólar')

# Exibir o gráfico
plt.show()


# In[29]:


# Carregar o dataframe com as cotações
df = dt_bt_dl

# Definir o tamanho do intervalo de média (por exemplo, 7 dias)
intervalo = '1M'

# Calcular a média dos preços em intervalos de tempo
df_media = df.resample(intervalo).mean()
df_media_outro_campo = df.resample(intervalo).mean()  

# Configurar o tamanho da figura
plt.figure(figsize=(10, 6))

# Plotar o gráfico de linhas da média dos preços
plt.plot(df_media.index, df_media['Dl_Fechamento'], marker='o', linestyle='-', color='b', label='Fechamento')
plt.plot(df_media.index, df_media_outro_campo['Dl_Abertura'], linestyle='--', color='r', label='Abertura')
plt.plot(df_media.index, df_media['Dl_Maior'], marker='o', linestyle='-', color='y', label='Maior')
plt.plot(df_media.index, df_media_outro_campo['Dl_Menor'], linestyle='--', color='g', label='Menor')

# Configurar os rótulos dos eixos
plt.xlabel('Data')
plt.ylabel('Média de Preço')

# Configurar o título do gráfico
plt.title('Média de Preço do dolar ao Longo do Periodo')
plt.legend(loc='lower right')
# Rotacionar os rótulos do eixo x para facilitar a leitura
plt.xticks(rotation=45)

# Exibir o gráfico
plt.show()


# In[30]:


# Carregar o dataframe com as cotações de criptomoedas
df = dt_bt_dl

plt.boxplot([df['Bt_Abertura'], df['Bt_Maior'], df['Bt_Menor'], df['Bt_Fechamento']])

# Definir os rótulos dos eixos x e y
plt.xticks([1,2,3,4], ['Bt_Fechamento','Bt_Abertura','Bt_Maior','Bt_Menor'])
plt.ylabel('Valor')

# Definir um título para o gráfico
plt.title('Boxplot dos Valores do BitCoin')

# Exibir o gráfico
plt.show()


# In[31]:


# Carregar o dataframe com as cotações de criptomoedas
df = dt_bt_dl

# Definir o tamanho do intervalo de média (por exemplo, 7 dias)
intervalo = '1M'

# Calcular a média dos preços em intervalos de tempo
df_media = df.resample(intervalo).mean()

# Configurar o tamanho da figura
plt.figure(figsize=(10, 6))

# Plotar o gráfico de linhas da média dos preços
plt.plot(df_media.index, df_media['Bt_Fechamento'], marker='o', linestyle='-', color='b', label='Fechamento')
plt.plot(df_media.index, df_media['Bt_Abertura'], linestyle='--', color='r', label='Abertura')
plt.plot(df_media.index, df_media['Bt_Maior'], marker='o', linestyle='-', color='y', label='Maior')
plt.plot(df_media.index, df_media['Bt_Menor'], linestyle='--', color='g', label='Menor')

# Configurar os rótulos dos eixos
plt.xlabel('Data')
plt.ylabel('Média de Preço')

# Configurar o título do gráfico
plt.title('Média de Preço do bitcoin ao Longo do Periodo')
plt.legend(loc='lower right')
# Rotacionar os rótulos do eixo x para facilitar a leitura
plt.xticks(rotation=45)

# Exibir o gráfico
plt.show()


# In[32]:


# Colocar campo data com index para unir dataframes
dtf_Eth = dtf_Eth.set_index('Data')

# junção dataframes ethereum e dolar
dt_et_dl=dtf_usd.join(dtf_Eth)


# In[33]:


# Armazenando o dataframe completo (desde 01/01/2020)
dt_et_dl_completo = dt_et_dl

# Filtrando os dados com inicio em 11-03-2020
dt_et_dl = dt_et_dl.loc['2020-03-11':]

dt_et_dl.info()


# In[34]:


dt_et_dl.isnull().sum()


# In[35]:


dt_et_dl.describe()


# In[36]:


# Carregar o dataframe com as cotações de criptomoedas
df = dt_et_dl

plt.boxplot([df['Et_Abertura'], df['Et_Maior'], df['Et_Menor'], df['Et_Fechamento']])

# Definir os rótulos dos eixos x e y
plt.xticks([1,2,3,4], ['Et_Fechamento','Et_Abertura','Et_Maior','Et_Menor'])
plt.ylabel('Valor')

# Definir um título para o gráfico
plt.title('Boxplot dos Valores do Ethereum')

# Exibir o gráfico
plt.show()


# In[37]:


# Carregar o dataframe com as cotações de criptomoedas
df = dt_et_dl

# Definir o tamanho do intervalo de média
intervalo = '1M'

# Calcular a média dos preços em intervalos de tempo
df_media = df.resample(intervalo).mean()

# Configurar o tamanho da figura
plt.figure(figsize=(10, 6))

# Plotar o gráfico de linhas da média dos preços
plt.plot(df_media.index, df_media['Et_Fechamento'], marker='o', linestyle='-', color='b', label='Fechamento')
plt.plot(df_media.index, df_media['Et_Abertura'], linestyle='--', color='r', label='Abertura')
plt.plot(df_media.index, df_media['Et_Maior'], marker='o', linestyle='-', color='y', label='Maior')
plt.plot(df_media.index, df_media['Et_Menor'], linestyle='--', color='g', label='Menor')

# Configurar os rótulos dos eixos
plt.xlabel('Data')
plt.ylabel('Média de Preço')

# Configurar o título do gráfico
plt.title('Média de Preço do Ethereum ao Longo do Periodo')
plt.legend(loc='lower right')
# Rotacionar os rótulos do eixo x para facilitar a leitura
plt.xticks(rotation=45)

# Exibir o gráfico
plt.show()


# In[38]:


print(dt_bt_dl)


# In[39]:


# Utilizando o dataframe com as cotações desde inicio 01/01/2020
df = dt_bt_dl_completo

# valor do bitcoin em reais
df['Btr_Fechamento'] = df['Bt_Fechamento'] * df['Dl_Fechamento']

# Calcule as métricas para dolar
df['Dl_Amplitude'] = df['Dl_Maior'] - df['Dl_Menor']  # Amplitude = Maior Valor - Menor Valor
df['Dl_Variacao'] = df['Dl_Fechamento'] - df['Dl_Abertura']  

# Calcule as métricas para bitcoin
df['Bt_Amplitude'] = df['Bt_Maior'] - df['Bt_Menor']  # Amplitude = Maior Valor - Menor Valor
df['Bt_Variacao'] = df['Bt_Fechamento'] - df['Bt_Abertura']

# Cálculo da média móvel simples de 50,100,200 dias do bitcoin em dolar
df['Bt_SMA 50'] = df['Bt_Fechamento'].rolling(window=50, min_periods=1).mean()
df['Bt_SMA 100'] = df['Bt_Fechamento'].rolling(window=100, min_periods=1).mean()
df['Bt_SMA 200'] = df['Bt_Fechamento'].rolling(window=200, min_periods=1).mean()

# Exemplo de cálculo do RSI de 14 dias
delta = df['Bt_Fechamento'].diff()  # Diferença entre os preços de fechamento consecutivos
gain = delta.where(delta > 0, 0)  # Ganho (diferença positiva)
loss = -delta.where(delta < 0, 0)  # Perda (diferença negativa)

average_gain = gain.rolling(window=14, min_periods=1).mean()  # Média móvel de ganhos de 14 dias
average_loss = loss.rolling(window=14, min_periods=1).mean()  # Média móvel de perdas de 14 dias
rs = average_gain / average_loss  # Rácio de ganhos/perdas
rsi = 100 - (100 / (1 + rs))  # RSI
df['rsi'] = rsi

# Cálculo da MACD
short_ema = df['Bt_Fechamento'].ewm(span=12, min_periods=1).mean()  # EMA de 12 dias
long_ema = df['Bt_Fechamento'].ewm(span=26, min_periods=1).mean()  # EMA de 26 dias
df['MACD'] = short_ema - long_ema  # MACD

# Cálculo das Bandas de Bollinger
df['SMA'] = df['Bt_Fechamento'].rolling(window=20, min_periods=1).mean()
df['Std'] = df['Bt_Fechamento'].rolling(window=20, min_periods=1).std()

# Tratamento do valor NAN do primeiro registro, usando proximo disponivel 
df['Std'].fillna(df['Std'].iloc[20 - 1], inplace=True)

df['BandaBollingerAlta'] = df['SMA'] + 2 * df['Std']
df['BandaBollingerBaixa'] = df['SMA'] - 2 * df['Std']

# Calcular os níveis de Fibonacci para cada dia
highest_high = df['Bt_Maior'].rolling(window=20, min_periods=1).max()
lowest_low = df['Bt_Menor'].rolling(window=20, min_periods=1).min()
range_high_low = highest_high - lowest_low

# Níveis de Retração de Fibonacci
fibonacci_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
for level in fibonacci_levels:
    df['Fibonacci Level ' + str(level)] = highest_high - level * range_high_low

# Definir o percentual de valorização e o número de dias a serem considerados
percentual_valorizacao = 0.10  
numero_dias = 7

# Calcular a valorização percentual em relação ao número de dias anteriores
df['Valorizacao'] = df['Bt_Fechamento'].pct_change(periods=numero_dias)

# Criar campo que indica se a moeda valorizou o percentual desejado nos dias futuros
df['Valorizou'] = (df['Valorizacao'].shift(-numero_dias) > percentual_valorizacao) & (df['Valorizacao'] > 0)
    
dt_bt_dl = df.loc['2020-03-11':]

# Filtrando os dados com inicio em 11-03-2020
print(dt_bt_dl)


# In[40]:


df[df['Valorizou'] == True].count().sum()


# In[41]:


# Utilizando o dataframe ethereum com as cotações desde inicio 01/01/2020
df = dt_et_dl_completo

# valor do bitcoin em reais
df['Etr_Fechamento'] = df['Et_Fechamento'] * df['Dl_Fechamento']

# Calcule as métricas para dolar
df['Dl_Amplitude'] = df['Dl_Maior'] - df['Dl_Menor']  # Amplitude = Maior Valor - Menor Valor
df['Dl_Variacao'] = df['Dl_Fechamento'] - df['Dl_Abertura']  

# Calcule as métricas para bitcoin
df['Et_Amplitude'] = df['Et_Maior'] - df['Et_Menor']  # Amplitude = Maior Valor - Menor Valor
df['Et_Variacao'] = df['Et_Fechamento'] - df['Et_Abertura']

# Cálculo da média móvel simples de 50,100,200 dias do bitcoin em dolar
df['Et_SMA 50'] = df['Et_Fechamento'].rolling(window=50, min_periods=1).mean()
df['Et_SMA 100'] = df['Et_Fechamento'].rolling(window=100, min_periods=1).mean()
df['Et_SMA 200'] = df['Et_Fechamento'].rolling(window=200, min_periods=1).mean()

# Exemplo de cálculo do RSI de 14 dias
delta = df['Et_Fechamento'].diff()  # Diferença entre os preços de fechamento consecutivos
gain = delta.where(delta > 0, 0)  # Ganho (diferença positiva)
loss = -delta.where(delta < 0, 0)  # Perda (diferença negativa)

average_gain = gain.rolling(window=14, min_periods=1).mean()  # Média móvel de ganhos de 14 dias
average_loss = loss.rolling(window=14, min_periods=1).mean()  # Média móvel de perdas de 14 dias
rs = average_gain / average_loss  # Rácio de ganhos/perdas
rsi = 100 - (100 / (1 + rs))  # RSI
df['rsi'] = rsi

# Cálculo da MACD
short_ema = df['Et_Fechamento'].ewm(span=12, min_periods=1).mean()  # EMA de 12 dias
long_ema = df['Et_Fechamento'].ewm(span=26, min_periods=1).mean()  # EMA de 26 dias
df['MACD'] = short_ema - long_ema  # MACD

# Cálculo das Bandas de Bollinger
df['SMA'] = df['Et_Fechamento'].rolling(window=20, min_periods=1).mean()
df['Std'] = df['Et_Fechamento'].rolling(window=20, min_periods=1).std()

# Tratamento do valor NAN do primeiro registro, usando proximo disponivel 
df['Std'].fillna(df['Std'].iloc[20 - 1], inplace=True)

df['BandaBollingerAlta'] = df['SMA'] + 2 * df['Std']
df['BandaBollingerBaixa'] = df['SMA'] - 2 * df['Std']

# Calcular os níveis de Fibonacci para cada dia
highest_high = df['Et_Maior'].rolling(window=20, min_periods=1).max()
lowest_low = df['Et_Menor'].rolling(window=20, min_periods=1).min()
range_high_low = highest_high - lowest_low

# Níveis de Retração de Fibonacci
fibonacci_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
for level in fibonacci_levels:
    df['Fibonacci Level ' + str(level)] = highest_high - level * range_high_low

# Definir o percentual de valorização e o número de dias a serem considerados
percentual_valorizacao = 0.10  
numero_dias = 7

# Calcular a valorização percentual em relação ao número de dias anteriores
df['Valorizacao'] = df['Et_Fechamento'].pct_change(periods=numero_dias)

# Criar campo que indica se a moeda valorizou o percentual desejado nos dias futuros
df['Valorizou'] = (df['Valorizacao'].shift(-numero_dias) > percentual_valorizacao) & (df['Valorizacao'] > 0)
    
dt_et_dl = df.loc['2020-03-11':]

# Filtrando os dados com inicio em 11-03-2020
print(dt_et_dl)


# In[42]:


df[df['Valorizou'] == True].count().sum()


# In[ ]:





# In[43]:


dt_bt_dl.isnull().sum()


# In[44]:


from sklearn.ensemble import ExtraTreesRegressor

# Carregar os dados da base de dados históricos
dados = dt_bt_dl

# Separar as variáveis de entrada (X) e a variável de saída (y)
X = dados.drop('Valorizou', axis=1)  # Valorizou é coluna target
y = dados['Valorizou']

# Criação do modelo de árvore de decisão para estimar a importância das variáveis
modelo = ExtraTreesRegressor()
modelo.fit(X, y)

# Extraindo a importância das variáveis
importancias = modelo.feature_importances_

# Criando um DataFrame para visualização das importâncias
importancias_df = pd.DataFrame({'Variável': X.columns, 'Importância': importancias})

# Ordenar as variáveis por importância descendente
importancias_df = importancias_df.sort_values('Importância', ascending=False)

# Exibir o resultado
print(importancias_df)


# In[45]:


from sklearn.ensemble import ExtraTreesRegressor

# Carregar os dados da base de dados históricos
dados = dt_et_dl

# Separar as variáveis de entrada (X) e a variável de saída (y)
X = dados.drop('Valorizou', axis=1)  # Valorizou é coluna target
y = dados['Valorizou']

# Criação do modelo de árvore de decisão para estimar a importância das variáveis
modelo = ExtraTreesRegressor()
modelo.fit(X, y)

# Extraindo a importância das variáveis
importancias = modelo.feature_importances_

# Criando um DataFrame para visualização das importâncias
importancias_df = pd.DataFrame({'Variável': X.columns, 'Importância': importancias})

# Ordenar as variáveis por importância descendente
importancias_df = importancias_df.sort_values('Importância', ascending=False)

# Exibir o resultado
print(importancias_df)


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# atribuir nossa base de cotaçoes de dolar
df = dt_bt_dl

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df[['Bt_SMA 200', 'rsi', 'Bt_SMA 100', 'Std', 'Dl_Amplitude', 'Bt_SMA 50', 'MACD','Fibonacci Level 0',
        'Dl_Menor', 'BandaBollingerAlta', 'Bt_Volume', 'Fibonacci Level 1', 'Dl_Maior']]
y = df['Valorizou']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliar desempenho usando as métricas de erro quadrático médio (MSE) e coeficiente de determinação (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Erro quadrático médio (MSE):", mse)
print("Coeficiente de determinação (R²):", r2)


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# atribuir nossa base de cotaçoes de ethereum
df = dt_et_dl

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df[['rsi', 'Et_SMA 50', 'MACD', 'Et_SMA 100', 'Et_Fechamento', 'Et_SMA 200', 'Std', 'Dl_Maior', 
'Et_Variacao', 'Fibonacci Level 0', 'Etr_Fechamento', 'Dl_Amplitude', 'Dl_Menor']]
y = df['Valorizou']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões com os dados de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo usando as métricas de erro quadrático médio (MSE) e coeficiente de determinação (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Erro quadrático médio (MSE):", mse)
print("Coeficiente de determinação (R²):", r2)


# In[62]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Carregando os dados
df = dt_bt_dl

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df[['Bt_SMA 200', 'rsi', 'Bt_SMA 100', 'Std', 'Dl_Amplitude', 'Bt_SMA 50', 'MACD','Fibonacci Level 0',
        'Dl_Menor', 'BandaBollingerAlta', 'Bt_Volume', 'Fibonacci Level 1', 'Dl_Maior']]
y = df['Valorizou']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação e treinamento do modelo de Árvore de Decisão
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Previsões com o conjunto de teste
predictions = model.predict(X_test)

# Avaliação do desempenho do modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy}")

# Matriz de Confusão
cm = confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(cm)

# Precisão
precision = precision_score(y_test, predictions)
print(f"Precisão: {precision}")

# Recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall}")

# F1-Score
f1 = f1_score(y_test, predictions)
print(f"F1-Score: {f1}")

# AUC-ROC
auc_roc = roc_auc_score(y_test, predictions)
print(f"AUC-ROC: {auc_roc}")


# In[63]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Carregando os dados
df = dt_et_dl

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df[['rsi', 'Et_SMA 50', 'MACD', 'Et_SMA 100', 'Et_Fechamento', 'Et_SMA 200', 'Std', 'Dl_Maior', 
'Et_Variacao', 'Fibonacci Level 0', 'Etr_Fechamento', 'Dl_Amplitude', 'Dl_Menor']]
y = df['Valorizou']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação e treinamento do modelo de Árvore de Decisão
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Previsões com o conjunto de teste
predictions = model.predict(X_test)

# Avaliação do desempenho do modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy}")

# Matriz de Confusão
cm = confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(cm)

# Precisão
precision = precision_score(y_test, predictions)
print(f"Precisão: {precision}")

# Recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall}")

# F1-Score
f1 = f1_score(y_test, predictions)
print(f"F1-Score: {f1}")

# AUC-ROC
auc_roc = roc_auc_score(y_test, predictions)
print(f"AUC-ROC: {auc_roc}")


# In[67]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregando os dados,ou seja, atribuir nossa base de cotaçoes de bitcoin
df = dt_bt_dl

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df[['Bt_SMA 200', 'rsi', 'Bt_SMA 100', 'Std', 'Dl_Amplitude', 'Bt_SMA 50', 'MACD','Fibonacci Level 0',
        'Dl_Menor', 'BandaBollingerAlta', 'Bt_Volume', 'Fibonacci Level 1', 'Dl_Maior']]
y = df['Valorizou']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação e treinamento do modelo Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Previsões com o conjunto de teste
predictions = model.predict(X_test)

print(f"Métricas obtidas da aplicação do modelo sobre")
print(f"as informações das cotações do Bitcoin")
print(f"")

# Avaliação do desempenho do modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy}")

# Matriz de Confusão
cm = confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(cm)

# Precisão
precision = precision_score(y_test, predictions)
print(f"Precisão: {precision}")

# Recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall}")

# F1-Score
f1 = f1_score(y_test, predictions)
print(f"F1-Score: {f1}")

# AUC-ROC
auc_roc = roc_auc_score(y_test, predictions)
print(f"AUC-ROC: {auc_roc}")


# In[68]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregando os dados,ou seja, atribuir nossa base de cotaçoes de etereum
df = dt_et_dl

# Separar as variáveis independentes (X) e a variável dependente (y)
X = df[['rsi', 'Et_SMA 50', 'MACD', 'Et_SMA 100', 'Et_Fechamento', 'Et_SMA 200', 'Std', 'Dl_Maior', 
'Et_Variacao', 'Fibonacci Level 0', 'Etr_Fechamento', 'Dl_Amplitude', 'Dl_Menor']]
y = df['Valorizou']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação e treinamento do modelo Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Previsões com o conjunto de teste
predictions = model.predict(X_test)

print(f"Métricas obtidas da aplicação do modelo sobre")
print(f"as informações das cotações do Ethereum")
print(f"")

# Avaliação do desempenho do modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia: {accuracy}")

# Matriz de Confusão
cm = confusion_matrix(y_test, predictions)
print("Matriz de Confusão:")
print(cm)

# Precisão
precision = precision_score(y_test, predictions)
print(f"Precisão: {precision}")

# Recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall}")

# F1-Score
f1 = f1_score(y_test, predictions)
print(f"F1-Score: {f1}")

# AUC-ROC
auc_roc = roc_auc_score(y_test, predictions)
print(f"AUC-ROC: {auc_roc}")


# In[ ]:






# In[ ]:




