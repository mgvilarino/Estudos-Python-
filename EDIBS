Fichero Python: agency_data_analysis.py
import pandas as pd
import numpy as np

def run_analysis():
    # 1. CARGA DE DATOS
    # Se asume que los ficheros CSV descargados de la ruta proporcionada 
    # mantienen los nombres de las fuentes analizadas.
    try:
        df_ads = pd.read_csv('Ads_Daily_Enhanced.csv') # [1]
        df_campaigns = pd.read_csv('Campaigns_Enhanced.csv') # [2]
        df_clients = pd.read_csv('Clients_Enhanced.csv') # [3]
        df_products = pd.read_csv('Products_Enhanced.csv') # [4]
        df_sales = pd.read_csv('Sales_Enhanced.csv') # [5]
    except FileNotFoundError as e:
        print(f"Error: Asegúrate de tener los archivos CSV en la misma carpeta. {e}")
        return

    # 2. MODELADO (Esquema Estrella)
    # Siguiendo las mejores prácticas, separamos Hechos de Dimensiones [8].
    # Unimos Ads (Hechos) con Campaigns (Dimensión) y Clients (Dimensión).
    df_performance = df_ads.merge(df_campaigns, on='campaign_id', how='left')
    df_performance = df_performance.merge(df_clients, on='client_id', how='left')

    # Agregamos las ventas por campaña para calcular ingresos totales
    df_sales_agg = df_sales.groupby('campaign_id')['revenue_usd'].sum().reset_index()
    
    # Unimos el rendimiento publicitario con el financiero
    df_master = df_performance.groupby(['client_name', 'channel']).agg({
        'ad_spend_usd': 'sum',
        'clicks': 'sum',
        'conversions': 'sum'
    }).reset_index()

    # Integración de ingresos (Revenue)
    # Nota: Se agrupa para asegurar que el ROAS sea preciso a nivel de segmento.
    campaign_revenue = df_campaigns.merge(df_sales_agg, on='campaign_id', how='left').fillna(0)
    client_revenue = campaign_revenue.merge(df_clients, on='client_id', how='left')
    client_totals = client_revenue.groupby(['client_name', 'channel'])['revenue_usd'].sum().reset_index()
    
    df_final = df_master.merge(client_totals, on=['client_name', 'channel'], how='left')

    # 3. CÁLCULO DE MÉTRICAS (Basado en r/PPC y Wall Street Prep)
    
    # ROAS (Return on Ad Spend) = Ingresos / Gasto [9, 10]
    df_final['ROAS'] = df_final['revenue_usd'] / df_final['ad_spend_usd']

    # CPA (Cost Per Acquisition) = Gasto / Conversiones [9]
    df_final['CPA'] = df_final['ad_spend_usd'] / df_final['conversions']

    # Tasa de Conversión = (Conversiones / Clicks) * 100 [11]
    df_final['Conv_Rate_Pct'] = (df_final['conversions'] / df_final['clicks']) * 100

    # 4. MEDIDAS PERSONALIZADAS (Nivel Avanzado)
    
    # LTV (Customer Lifetime Value) Estimado [12, 13]
    # Fórmula: (ARPA * Margen Bruto) / Churn Rate
    # Valores asuntivos para el ejemplo (80% margen, 2.5% churn mensual)
    margin = 0.80
    churn = 0.025
    df_final['ARPA'] = df_final['revenue_usd'] # Ingreso promedio asumido
    df_final['Est_CLV'] = (df_final['ARPA'] * margin) / churn

    # ROAS Dinámico considerando el Retainer de la agencia [3]
    # Se obtiene el retainer sumándolo al gasto publicitario para ver rentabilidad total
    retainers = df_clients[['client_name', 'monthly_retainer_usd']].drop_duplicates()
    df_final = df_final.merge(retainers, on='client_name', how='left')
    df_final['ROAS_Total_Account'] = df_final['revenue_usd'] / (df_final['ad_spend_usd'] + df_final['monthly_retainer_usd'])

    # 5. RESULTADOS
    print("--- Análisis de Rendimiento por Cliente y Canal ---")
    print(df_final[['client_name', 'channel', 'ROAS', 'CPA', 'Conv_Rate_Pct', 'Est_CLV']].round(2))

    # Guardar el modelo procesado
    df_final.to_csv('Agencia_Star_Schema_Fact.csv', index=False)
    print("\nFichero 'Agencia_Star_Schema_Fact.csv' generado exitosamente.")

if __name__ == "__main__":
    run_analysis()
