# main.py - VERSÃO FINAL PARA APRESENTAÇÃO DE BIG DATA (V3.0)

# ===============================================================
# INSTALAÇÃO E IMPORTAÇÕES
# ===============================================================

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize

# --- IMPORTAÇÕES PARA MACHINE LEARNING ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# --- CORREÇÃO DO ERRO NLTK ---
def download_nltk_resources():
    """Baixa recursos necessários do NLTK, incluindo 'punkt' e 'punkt_tab'."""
    resources = ['punkt', 'punkt_tab']
    
    print("\n--- Verificando recursos do NLTK ---")
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Baixando recurso '{resource}'...")
            try:
                nltk.download(resource, quiet=True)
                print(f"Download de '{resource}' concluído.")
            except Exception as e:
                print(f"Erro ao baixar '{resource}': {e}. Tente 'python -m nltk.downloader {resource}' no terminal.")

download_nltk_resources()


# Configuração de visualização para Matplotlib
plt.style.use('ggplot')
sns.set_palette("tab10")

# ===============================================================
# ETAPAS 1, 2 e 3 (Ingestão, Limpeza e Vetorização de Dados Estruturados)
# ===============================================================

print("\n--- 1. INGESTÃO DE DADOS ---")
FILE_2024 = "datatran2024.csv"
FILE_2025 = "datatran2025.csv"

try:
    df_2024 = pd.read_csv(FILE_2024, sep=';', encoding='latin-1')
    df_2025 = pd.read_csv(FILE_2025, sep=';', encoding='latin-1')
except Exception:
    df_2024 = pd.read_csv(FILE_2024, sep=';', encoding='utf-8')
    df_2025 = pd.read_csv(FILE_2025, sep=';', encoding='utf-8')

df_total = pd.concat([df_2024, df_2025], ignore_index=True)

# 2. REFINAMENTO
cols_geo = ['latitude', 'longitude']
for col in cols_geo:
    df_total[col] = df_total[col].astype(str).str.replace(',', '.', regex=False)
    df_total[col] = pd.to_numeric(df_total[col], errors='coerce')

df_geo = df_total.dropna(subset=cols_geo).copy()
df_geo['data_inversa'] = pd.to_datetime(df_geo['data_inversa'], format='%Y-%m-%d', errors='coerce')

# 3. VETORIZAÇÃO (FEATURES)
df_geo['ano'] = df_geo['data_inversa'].dt.year
df_geo['mes'] = df_geo['data_inversa'].dt.month

# Agregação para gráficos
acidentes_por_uf = df_geo['uf'].value_counts().sort_values(ascending=False).reset_index(name='Total_Acidentes')
acidentes_por_uf.columns = ['UF', 'Total_Acidentes']

top_causas = df_geo['causa_acidente'].value_counts().head(10).reset_index(name='Total_Acidentes')
top_causas.columns = ['Causa_Acidente', 'Total_Acidentes']

print(f"Total de registros processados: {len(df_geo):,}".replace(",", "."))
print("Etapas 1, 2 e 3 (Estruturadas) concluídas.")

# ===============================================================
# ETAPA 4: PROCESSAMENTO DE DADOS NÃO ESTRUTURADOS E PREDIÇÃO
# ===============================================================

# --- 4.1. WEB MINING E NLP (SIMULADO) ---
def web_mining_g1(query):
    """Simula a busca e processamento de notícias com NLP."""
    simulated_title = f"G1: Acidente grave em {query} deixa feridos e causa lentidão na BR-101."
    simulated_body = "O acidente ocorreu na madrugada desta quarta-feira, envolvendo três veículos. A causa provável foi a ausência de reação do condutor. A Polícia Rodoviária Federal confirmou o óbito de uma pessoa e ferimentos graves em outras duas."
    
    tokens_body = word_tokenize(simulated_body, language='portuguese')
    
    termos_foco = [t.lower() for t in tokens_body if t.lower() in ['acidente', 'mortos', 'feridos', 'óbito', 'colisão', 'velocidade', 'chuva']]
    
    print(f"\n- Busca Web Mining Simulada: '{query}'")
    print(f"  - Termos de Foco (NLP): {', '.join(set(termos_foco))}")
    
    return termos_foco

print("\n--- 4.1. MÓDULO DE WEB MINING (G1 + NLP) ---")

top_3_causas = df_geo['causa_acidente'].value_counts().head(3).index.tolist()

all_extracted_terms = []
for causa in top_3_causas:
    extracted_terms = web_mining_g1(causa)
    all_extracted_terms.extend(extracted_terms)

if all_extracted_terms:
    freq_dist = nltk.FreqDist(all_extracted_terms)
    print("\nFrequência dos Termos-Chave (Validação dos Dados Não Estruturados):")
    for word, frequency in freq_dist.most_common(5):
        print(f"  - {word.capitalize()}: {frequency}")


# --- 4.2. MODELAGEM PREDITIVA: Previsão de Severidade ---
print("\n--- 4.2. MODELAGEM PREDITIVA: Previsão de Severidade ---")

# 1. Definir a Variável Alvo (Severidade Alta)
df_model = df_geo.copy()
df_model['severidade_alta'] = np.where(
    (df_model['mortos'] > 0) | (df_model['feridos_graves'] > 0), 1, 0
)

# 2. Seleção e Vetorização de Features (One-Hot Encoding)
features = ['causa_acidente', 'tipo_acidente', 'condicao_metereologica', 'tipo_pista', 'uf', 'fase_dia']
df_features = pd.get_dummies(df_model[features], drop_first=True)

X = df_features
y = df_model['severidade_alta']

if X.shape[0] > 100 and y.sum() > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. Treinamento do Modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Avaliação
    y_pred = model.predict(X_test)
    
    print("\nRelatório de Classificação da Severidade do Acidente:")
    print(classification_report(y_test, y_pred))

    # 6. Extração da Importância das Features
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_10_features = feature_importances.nlargest(10)
    
    print("\nTop 10 Fatores Mais Importantes na Previsão de Severidade:")
    print(top_10_features)

else:
    print("Dados insuficientes ou desbalanceados para treinamento de modelo preditivo.")
    top_10_features = None # Define como None para não quebrar o gráfico

print("Modelagem Preditiva concluída.")

# ===============================================================
# ETAPA 5: ANÁLISE E VISUALIZAÇÃO (TODOS OS GRÁFICOS)
# ===============================================================

print("\n--- 5. GERAÇÃO DE GRÁFICOS E MAPA ---")

# Define o gradiente de cores (reutilizado em todos os mapas)
custom_gradient = {0.0: 'green', 0.5: 'yellow', 1.0: 'red'}

# Define o HTML da legenda (reutilizado em todos os mapas)
legend_html_base = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 180px; height: 100px; 
                 border:1px solid grey; z-index:9999; font-size:14px;
                 background-color: white; opacity: 0.95; padding: 5px;">
       &nbsp; <b>Densidade de Acidentes</b> <br>
       &nbsp; <i style="background: red; width: 10px; height: 10px; display: inline-block; margin-right: 5px;"></i> Alto (Muitos) <br>
       &nbsp; <i style="background: yellow; width: 10px; height: 10px; display: inline-block; margin-right: 5px;"></i> Médio (Intermediário) <br>
       &nbsp; <i style="background: green; width: 10px; height: 10px; display: inline-block; margin-right: 5px;"></i> Baixo (Menos)
     </div>
     '''


# 5.1. MAPA DE CALOR GEOGRÁFICO - GLOBAL
heatmap_data = df_geo[['latitude', 'longitude']].values.tolist()
lat_br, lon_br = -15.7797200, -47.9297200
mapa = folium.Map(location=[lat_br, lon_br], zoom_start=4)

HeatMap(heatmap_data, radius=8, max_zoom=13, gradient=custom_gradient).add_to(mapa)

legend_html_global = legend_html_base.replace("Densidade de Acidentes", "Densidade de Acidentes (GLOBAL)")
mapa.get_root().html.add_child(folium.Element(legend_html_global))

mapa_file = 'mapa_calor_acidentes_global.html'
mapa.save(mapa_file)
print(f"Mapa de Calor Global salvo em: {mapa_file}")


# 5.9. NOVO MAPA DE CALOR ESPECÍFICO: ESTADO DO PARANÁ (PR)
df_parana = df_geo[df_geo['uf'] == 'PR'].copy()
heatmap_data_pr = df_parana[['latitude', 'longitude']].values.tolist()

lat_pr_center = df_parana['latitude'].mean() if not df_parana.empty else -25.0
lon_pr_center = df_parana['longitude'].mean() if not df_parana.empty else -51.5

mapa_pr = folium.Map(location=[lat_pr_center, lon_pr_center], zoom_start=7)

HeatMap(heatmap_data_pr, radius=10, max_zoom=13, gradient=custom_gradient).add_to(mapa_pr)

legend_html_pr = legend_html_base.replace("Densidade de Acidentes", "Densidade de Acidentes (PR)")
mapa_pr.get_root().html.add_child(folium.Element(legend_html_pr))

mapa_parana_file = 'mapa_calor_parana.html'
mapa_pr.save(mapa_parana_file)
print(f"Mapa de Calor do Paraná (PR) salvo em: {mapa_parana_file}")


# 5.2. GRÁFICO DE BARRAS: Acidentes por UF
plt.figure(figsize=(12, 6))
sns.barplot(x='UF', y='Total_Acidentes', data=acidentes_por_uf, palette='viridis', hue='UF', legend=False)
plt.title('Total de Acidentes por Estado (UF) - 2024/2025', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('grafico_acidentes_por_uf.png')

# 5.3. GRÁFICO DE BARRAS: Top 10 Causas de Acidente
plt.figure(figsize=(14, 7))
sns.barplot(x='Total_Acidentes', y='Causa_Acidente', data=top_causas, palette='inferno', hue='Causa_Acidente', legend=False)
plt.title('Top 10 Causas de Acidente - 2024/2025', fontsize=16)
plt.xlabel('Total de Acidentes', fontsize=12)
plt.ylabel('Causa do Acidente', fontsize=12)
plt.tight_layout()
plt.savefig('grafico_top_causas.png')

# 5.4. GRÁFICO ADICIONAL: Acidentes por Fase do Dia
fase_dia_counts = df_geo['fase_dia'].value_counts().reset_index(name='Total')
fase_dia_counts.columns = ['Fase_Dia', 'Total']
plt.figure(figsize=(10, 6))
sns.barplot(x='Fase_Dia', y='Total', data=fase_dia_counts, palette='Paired', hue='Fase_Dia', legend=False)
plt.title('Acidentes por Fase do Dia', fontsize=16)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('grafico_fase_dia.png')

# 5.5. GRÁFICO ADICIONAL: Distribuição de Mortes e Feridos
impact_data = df_geo[['mortos', 'feridos_graves', 'feridos_leves']].sum().reset_index(name='Total')
impact_data.columns = ['Tipo_Vitima', 'Total']
plt.figure(figsize=(8, 6))
sns.barplot(x='Tipo_Vitima', y='Total', data=impact_data, palette=['darkred', 'orange', 'gold'], hue='Tipo_Vitima', legend=False)
plt.title('Impacto dos Acidentes (Mortos vs. Feridos)', fontsize=16)
plt.tight_layout()
plt.savefig('grafico_impacto_vitimas.png')

# 5.6. GRÁFICO ADICIONAL: Acidentes por Mês (Sazonalidade)
sazonalidade = df_geo.groupby(['ano', 'mes']).size().reset_index(name='Total')
sazonalidade['Data'] = sazonalidade['ano'].astype(str) + '-' + sazonalidade['mes'].astype(str).str.zfill(2)
plt.figure(figsize=(14, 7))
sns.lineplot(x='Data', y='Total', hue='ano', data=sazonalidade, marker='o')
plt.title('Sazonalidade de Acidentes por Mês (2024 vs 2025)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Ano')
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_sazonalidade_mes.png')


# 5.7. SEVERIDADE POR TIPO DE ACIDENTE (Gráfico de Barras Empilhadas)
top_tipos_acidente = df_geo['tipo_acidente'].value_counts().head(8).index.tolist()
df_severidade = df_geo[df_geo['tipo_acidente'].isin(top_tipos_acidente)].copy()

df_severidade_agg = df_severidade.groupby('tipo_acidente')[['mortos', 'feridos_graves', 'feridos_leves']].sum()
df_severidade_agg = df_severidade_agg.sort_values(by='mortos', ascending=False)

plt.figure(figsize=(14, 8))
df_severidade_agg[['feridos_leves', 'feridos_graves', 'mortos']].plot(
    kind='barh', 
    stacked=True, 
    figsize=(14, 8), 
    color=['gold', 'orange', 'darkred'],
    ax=plt.gca()
)
plt.title('Severidade de Vítimas nos 8 Principais Tipos de Acidente', fontsize=16)
plt.xlabel('Número Total de Vítimas (2024/2025)', fontsize=12)
plt.ylabel('Tipo de Acidente', fontsize=12)
plt.legend(title='Severidade', labels=['Feridos Leves', 'Feridos Graves', 'Mortos'])
plt.tight_layout()
plt.savefig('grafico_severidade_por_tipo.png')

# 5.8. ACIDENTES POR CONDIÇÃO METEREOLÓGICA
acidentes_por_clima = df_geo['condicao_metereologica'].value_counts().sort_values(ascending=False).reset_index(name='Total')
acidentes_por_clima.columns = ['Condicao_Metereologica', 'Total']

plt.figure(figsize=(12, 7))
sns.barplot(x='Total', y='Condicao_Metereologica', data=acidentes_por_clima, palette='Blues_d', hue='Condicao_Metereologica', legend=False)
plt.title('Total de Acidentes por Condição Metereológica', fontsize=16)
plt.xlabel('Total de Acidentes', fontsize=12)
plt.ylabel('Condição Metereológica', fontsize=12)
plt.tight_layout()
plt.savefig('grafico_acidentes_por_clima.png')


# 5.10. NOVO GRÁFICO: IMPORTÂNCIA DE FEATURES DO MODELO PREDITIVO
# --------------------------------------------------------------------------------
if top_10_features is not None and not top_10_features.empty:
    plt.figure(figsize=(14, 8))
    # Para visualização, limpamos os nomes das features One-Hot-Encoded
    labels = [s.replace('tipo_acidente_', '').replace('condicao_metereologica_', '').replace('causa_acidente_', '').replace('_', ' ').replace('fase_dia_', '')
              for s in top_10_features.index]
    
    sns.barplot(x=top_10_features.values, y=labels, palette='magma')
    plt.title('Top 10 Fatores Mais Importantes na Previsão de Severidade (ML)', fontsize=16)
    plt.xlabel('Importância (Valor GINI)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('grafico_importancia_features_ml.png')
    print("Gráfico de Importância de Features (ML) salvo em: grafico_importancia_features_ml.png")
# --------------------------------------------------------------------------------


# Mensagem final
print("\n" + "="*70)
print("PROCESSAMENTO FINALIZADO. TRABALHO DE BIG DATA CONCLUÍDO.")
print("Total de 10 arquivos gerados:")
print("- 2 Mapas de Calor (HTML) e 8 Gráficos de Análise (PNG).")
print("======================================================================")