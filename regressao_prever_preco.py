import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from datetime import date

class PrevendoPreco():

    def __init__(self, acaoEscolhida):
        
        self.acaoEscolhida = acaoEscolhida

    def pegando_dados(self):

        cotacoes = pd.read_parquet(r'C:\Users\Caio\Documents\dev\github\regressao_prever_preco\dados\cotacoes.parquet')
        
        dados = cotacoes[cotacoes['ticker'] == self.acaoEscolhida]

        self.datas = pd.to_datetime(dados['data'].iloc[:-1]).dt.date
        
        dados = dados[['preco_fechamento_ajustado', 'volume_negociado']]
        
        self.dados = dados

    def criando_y(self):

        dados = self.dados

        dados['cotacao_dia_seguinte'] = dados['preco_fechamento_ajustado'].shift(-1)
        dados = dados.dropna()

        self.tamanhoDadosTreinamento = int(len(dados) * 0.8)

        self.escaladorTreinamento = MinMaxScaler(feature_range=(0, 1))
        self.escaladorTeste = MinMaxScaler(feature_range=(0, 1))

        dadosEntreZeroEUmTreinamento = self.escaladorTreinamento.fit_transform(dados.iloc[0: self.tamanhoDadosTreinamento, :])
        
        dadosEntreZeroEUmTeste = self.escaladorTeste.fit_transform(dados.iloc[self.tamanhoDadosTreinamento: , :])

        self.xTreinamento = dadosEntreZeroEUmTreinamento[:,:2]
        self.yTreinamento = dadosEntreZeroEUmTreinamento[:,2:]

        self.xTeste = dadosEntreZeroEUmTeste[:,:2]
        self.yTeste = dadosEntreZeroEUmTeste[:,2:]

    def criando_modelo_regressao(self):

        regressaoLinear = LinearRegression()
        regressaoLinear.fit(self.xTreinamento, self.yTreinamento)

        precosPreditos = regressaoLinear.predict(self.xTeste)

        self.scoreTreinamento = regressaoLinear.score(self.xTreinamento, self.yTreinamento)
        self.scoreTeste = regressaoLinear.score(self.xTeste, self.yTeste)

        dadosTeste = np.concatenate((self.xTeste, self.yTeste), axis=1)
        dadosPreditos = np.concatenate((self.xTeste, precosPreditos), axis=1)

        self.precosTesteReais = self.escaladorTeste.inverse_transform(dadosTeste)
        self.precosTestePreditos = self.escaladorTeste.inverse_transform(dadosPreditos)

    def grafico_predicao(self):

        fig, ax = plt.subplots(figsize= (10,4))

        ax.plot(self.datas.iloc[self.tamanhoDadosTreinamento:], self.precosTesteReais[:, 2], label= 'acao')
        ax.plot(self.datas.iloc[self.tamanhoDadosTreinamento:], self.precosTestePreditos[:, 2], label= 'predicao')

        plt.legend()
        plt.savefig('grafico_predicao.png', dpi= 300)
        plt.close()

    def avaliando_predicao(self):
        
        df = pd.DataFrame(self.precosTestePreditos, index= self.datas.iloc[self.tamanhoDadosTreinamento:])
        df.columns = ['preco', 'volume', 'preco_predito_dia_seguinte']

        df['retorno'] = df['preco'].pct_change()
        
        df['comprado_vendido'] = pd.NA

        df.loc[df['preco_predito_dia_seguinte'] > df['preco'], 'comprado_vendido'] = 'comprado'
        df.loc[df['preco_predito_dia_seguinte'] < df['preco'], 'comprado_vendido'] = 'vendido'

        df['acertos'] = pd.NA

        df.loc[(df['comprado_vendido'] == 'comprado') & (df['retorno'] > 0), 'acertos'] = 1
        df.loc[(df['comprado_vendido'] == 'comprado') & (df['retorno'] < 0), 'acertos'] = 0
        df.loc[(df['comprado_vendido'] == 'vendido') & (df['retorno'] > 0), 'acertos'] = 0
        df.loc[(df['comprado_vendido'] == 'vendido') & (df['retorno'] < 0), 'acertos'] = 1
        df.loc[df['acertos'].isna(), 'acertos'] = 0

        df = df.dropna()

        acertou = df['acertos'].sum()/len(df)
        errou = 1 - acertou

        df['retorno_absoluto'] = df['retorno'].abs()

        mediaLucrosEPerdas = df.groupby('acertos')['retorno_absoluto'].mean()

        self.expectativaMatematicaLucro = acertou * mediaLucrosEPerdas[1] - mediaLucrosEPerdas[0] * errou

        df['retorno_modelo'] = pd.NA

        df.loc[df['acertos'] == True, 'retorno_modelo'] = df.loc[df['acertos'] == True]['retorno_absoluto']
        df.loc[df['acertos'] == False, 'retorno_modelo'] = df.loc[df['acertos'] == False]['retorno_absoluto'] * - 1

        df['retorno_acum_modelo'] = (1 + df['retorno_modelo']).cumprod() - 1
        df['retorno_acum_acao'] = (1 + df['retorno']).cumprod() - 1

        self.df = df
        self.retornos = df[['retorno_acum_modelo', 'retorno_acum_acao']]

    def grafico_retorno(self):

        fig, ax = plt.subplots(figsize= (10,4))

        ax.plot(self.retornos.iloc[:,0], label= 'retorno_acum_modelo')
        ax.plot(self.retornos.iloc[:,1], label= 'retorno_acum_acao')

        plt.legend()
        plt.savefig('retorno_acumulado.png', dpi= 300)
        plt.close()


if __name__ == '__main__':

    predicao = PrevendoPreco(acaoEscolhida= 'VALE3')

    predicao.pegando_dados()
    predicao.criando_y()
    predicao.criando_modelo_regressao()
    predicao.grafico_predicao()
    predicao.avaliando_predicao()
    predicao.grafico_retorno()

    # print(predicao.retornos)
    # print(predicao.df)