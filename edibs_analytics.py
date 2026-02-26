"""
EDIBS School — Analytics Pipeline
===================================
Carga los 5 datasets desde Google Drive, construye el esquema estrella
y calcula todas las métricas clave: ROAS, CPA, CVR, CPA ajustado, LTV.

Requisitos:
    pip install pandas gdown openpyxl

Fuente:
    Google Drive folder: https://drive.google.com/drive/u/0/folders/1UF-GL2f1f8dAMTbZpAkcbM4NAV2uybUS
"""

import io
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN DE FUENTES (Google Drive)
# ─────────────────────────────────────────────

# Carpeta raíz del proyecto
DRIVE_FOLDER_URL = "https://drive.google.com/drive/u/0/folders/1UF-GL2f1f8dAMTbZpAkcbM4NAV2uybUS"

# IDs individuales de cada archivo dentro de la carpeta.
# Para obtenerlos: clic derecho sobre el archivo en Drive → "Obtener enlace"
# El ID es la parte tras /d/ en la URL:  .../d/<FILE_ID>/view
# ⚠️ Reemplaza los valores de FILE_ID con los IDs reales una vez que los archivos
#    estén compartidos con acceso público ("Cualquiera con el enlace").
FILE_IDS = {
    "sales":     "SALES_FILE_ID_AQUI",
    "products":  "PRODUCTS_FILE_ID_AQUI",
    "clients":   "CLIENTS_FILE_ID_AQUI",
    "campaigns": "CAMPAIGNS_FILE_ID_AQUI",
    "ads":       "ADS_DAILY_FILE_ID_AQUI",
}

# URL de descarga directa para archivos CSV públicos en Google Drive
def drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ─────────────────────────────────────────────
# 2. CARGA DE DATOS
# ─────────────────────────────────────────────

def load_dataframes() -> dict[str, pd.DataFrame]:
    """
    Descarga y carga los CSV desde Google Drive.
    Retorna un diccionario con los DataFrames ya tipados.
    """
    print("📥 Cargando datos desde Google Drive...")
    print(f"   Carpeta: {DRIVE_FOLDER_URL}\n")

    dfs = {}
    dtype_map = {
        "sales":     {"campaign_id": int, "product_id": int, "revenue_usd": float},
        "products":  {"product_id": int, "price_usd": float},
        "clients":   {"client_id": int, "monthly_retainer_usd": float},
        "campaigns": {"campaign_id": int, "client_id": int, "budget_usd": float},
        "ads":       {
            "campaign_id": int,
            "impressions": int,
            "clicks": int,
            "conversions": int,
            "ad_spend_usd": float,
        },
    }
    date_cols = {
        "sales":     ["date"],
        "campaigns": ["start_date", "end_date"],
        "ads":       ["date"],
    }

    for name, file_id in FILE_IDS.items():
        url = drive_url(file_id)
        try:
            df = pd.read_csv(
                url,
                dtype=dtype_map.get(name),
                parse_dates=date_cols.get(name),
            )
            dfs[name] = df
            print(f"   ✅ {name:12s}  →  {df.shape[0]:,} filas × {df.shape[1]} cols")
        except Exception as exc:
            print(f"   ❌ {name:12s}  →  Error al cargar: {exc}")
            raise

    print()
    return dfs


# ─────────────────────────────────────────────
# 3. CONSTRUCCIÓN DEL ESQUEMA ESTRELLA
# ─────────────────────────────────────────────

def build_star_schema(dfs: dict) -> dict[str, pd.DataFrame]:
    """
    Construye las tablas de hechos y dimensiones del esquema estrella.

    Esquema:
        FACT_AD_PERFORMANCE  ──→  DIM_CAMPAIGN
                             ──→  DIM_DATE
                             ──→  DIM_PRODUCT
        DIM_CAMPAIGN         ──→  DIM_CLIENT
                             ──→  DIM_CHANNEL
    """
    print("🏗️  Construyendo esquema estrella...")

    # ── Dimensiones ──────────────────────────────

    # DIM_CLIENT
    dim_client = dfs["clients"].copy().rename(columns={"monthly_retainer_usd": "monthly_retainer"})
    dim_client["created_at"] = pd.Timestamp.now()

    # DIM_CHANNEL (normalizada desde campaigns)
    dim_channel = (
        dfs["campaigns"][["channel"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"channel": "channel_name"})
    )
    dim_channel.insert(0, "channel_id", range(1, len(dim_channel) + 1))

    # DIM_CAMPAIGN (con channel_id como FK)
    dim_campaign = dfs["campaigns"].copy().merge(
        dim_channel, left_on="channel", right_on="channel_name", how="left"
    ).drop(columns=["channel"])

    # DIM_PRODUCT
    dim_product = dfs["products"].copy()
    dim_product["margin_pct"] = 0.40  # margen estimado; reemplazar con dato real

    # DIM_DATE (calendario completo del período)
    all_dates = pd.concat([
        dfs["ads"]["date"],
        dfs["sales"]["date"],
    ]).drop_duplicates().sort_values()
    dim_date = pd.DataFrame({"full_date": all_dates})
    dim_date["date_id"]     = range(1, len(dim_date) + 1)
    dim_date["year"]        = dim_date["full_date"].dt.year
    dim_date["quarter"]     = dim_date["full_date"].dt.quarter
    dim_date["month"]       = dim_date["full_date"].dt.month
    dim_date["month_name"]  = dim_date["full_date"].dt.month_name()
    dim_date["week"]        = dim_date["full_date"].dt.isocalendar().week.astype(int)
    dim_date["day_of_week"] = dim_date["full_date"].dt.day_name()
    dim_date["is_weekend"]  = dim_date["full_date"].dt.dayofweek >= 5

    # ── Tabla de hechos principal ────────────────

    # Agrego revenue de Sales a nivel (campaign_id, date)
    sales_agg = (
        dfs["sales"]
        .groupby(["campaign_id", "date"])["revenue_usd"]
        .sum()
        .reset_index()
    )

    fact = (
        dfs["ads"]
        .merge(sales_agg, on=["campaign_id", "date"], how="left")
        .merge(dim_date[["full_date", "date_id"]], left_on="date", right_on="full_date", how="left")
        .merge(dim_campaign[["campaign_id", "client_id", "channel_id"]], on="campaign_id", how="left")
        .drop(columns=["date", "full_date"])
        .fillna({"revenue_usd": 0})
    )
    fact.insert(0, "fact_id", range(1, len(fact) + 1))

    # ── Tabla de hechos de ventas ────────────────
    fact_sales = (
        dfs["sales"]
        .merge(dim_date[["full_date", "date_id"]], left_on="date", right_on="full_date", how="left")
        .merge(dim_campaign[["campaign_id", "client_id"]], on="campaign_id", how="left")
        .drop(columns=["date", "full_date"])
    )
    fact_sales.insert(0, "sale_id", range(1, len(fact_sales) + 1))

    schema = {
        "fact_ad_performance": fact,
        "fact_sales":          fact_sales,
        "dim_campaign":        dim_campaign,
        "dim_client":          dim_client,
        "dim_channel":         dim_channel,
        "dim_product":         dim_product,
        "dim_date":            dim_date,
    }

    for tname, tdf in schema.items():
        print(f"   📋 {tname:25s}  {tdf.shape[0]:,} filas × {tdf.shape[1]} cols")
    print()
    return schema


# ─────────────────────────────────────────────
# 4. CÁLCULO DE MÉTRICAS
# ─────────────────────────────────────────────

def compute_metrics(schema: dict) -> dict[str, pd.DataFrame]:
    """
    Calcula ROAS, CPA, CVR, CTR, CPA ajustado y LTV estimado.
    Retorna un diccionario con DataFrames listos para reportar o exportar.
    """
    print("📐 Calculando métricas...")

    fact    = schema["fact_ad_performance"]
    clients = schema["dim_client"]
    channel = schema["dim_channel"]
    camps   = schema["dim_campaign"]
    date    = schema["dim_date"]

    # ── Métricas base a nivel de fila ────────────
    fact = fact.copy()
    fact["ROAS"] = np.where(fact["ad_spend_usd"] > 0,
                            fact["revenue_usd"] / fact["ad_spend_usd"], np.nan)
    fact["CPA"]  = np.where(fact["conversions"] > 0,
                            fact["ad_spend_usd"] / fact["conversions"], np.nan)
    fact["CVR"]  = np.where(fact["clicks"] > 0,
                            fact["conversions"] / fact["clicks"], np.nan)
    fact["CTR"]  = np.where(fact["impressions"] > 0,
                            fact["clicks"] / fact["impressions"], np.nan)

    # ── ROAS dinámico por cliente ─────────────────
    client_agg = (
        fact
        .merge(camps[["campaign_id", "client_id"]], on="campaign_id", how="left")
        .groupby("client_id")
        .agg(
            total_revenue     = ("revenue_usd",  "sum"),
            total_spend       = ("ad_spend_usd", "sum"),
            total_conversions = ("conversions",  "sum"),
            total_clicks      = ("clicks",       "sum"),
            total_impressions = ("impressions",  "sum"),
        )
        .reset_index()
        .merge(clients[["client_id", "client_name", "industry", "monthly_retainer"]], on="client_id")
    )
    client_agg["ROAS_dinamico"] = (client_agg["total_revenue"] / client_agg["total_spend"]).round(2)
    client_agg["CPA"]           = (client_agg["total_spend"]   / client_agg["total_conversions"]).round(2)
    client_agg["CVR_pct"]       = (client_agg["total_conversions"] / client_agg["total_clicks"] * 100).round(2)
    client_agg["CTR_pct"]       = (client_agg["total_clicks"]  / client_agg["total_impressions"] * 100).round(2)
    # LTV estimado: valor medio de pedido × frecuencia de compra estimada (3.5)
    client_agg["avg_order_value"] = (client_agg["total_revenue"] / client_agg["total_conversions"]).round(2)
    client_agg["LTV_estimado"]    = (client_agg["avg_order_value"] * 3.5).round(2)

    # ── CPA ajustado por canal (escala de audiencia) ──
    channel_agg = (
        fact
        .merge(camps[["campaign_id", "channel_id"]], on="campaign_id", how="left")
        .groupby("channel_id")
        .agg(
            total_revenue     = ("revenue_usd",  "sum"),
            total_spend       = ("ad_spend_usd", "sum"),
            total_conversions = ("conversions",  "sum"),
            total_clicks      = ("clicks",       "sum"),
            total_impressions = ("impressions",  "sum"),
        )
        .reset_index()
        .merge(channel, on="channel_id")
    )
    channel_agg["ROAS"]     = (channel_agg["total_revenue"] / channel_agg["total_spend"]).round(2)
    channel_agg["CPA_neto"] = (channel_agg["total_spend"]   / channel_agg["total_conversions"]).round(2)
    channel_agg["CVR_pct"]  = (channel_agg["total_conversions"] / channel_agg["total_clicks"] * 100).round(2)
    # CPA ajustado: penaliza canales con mayor escala de impresiones (audiencia)
    avg_imp = channel_agg["total_impressions"].mean()
    channel_agg["CPA_ajustado"] = (
        channel_agg["CPA_neto"] * (channel_agg["total_impressions"] / avg_imp)
    ).round(2)

    # ── Costo marginal por conversión (semanal) ───
    fact_with_date = fact.merge(date[["date_id", "week", "year"]], on="date_id", how="left")
    weekly = (
        fact_with_date
        .groupby(["year", "week"])
        .agg(spend=("ad_spend_usd", "sum"), conversions=("conversions", "sum"))
        .reset_index()
        .sort_values(["year", "week"])
    )
    weekly["marginal_cost"] = (weekly["spend"].diff() / weekly["conversions"].diff()).round(2)

    # ── Evolución mensual ─────────────────────────
    fact_with_date2 = fact.merge(date[["date_id", "year", "month", "month_name"]], on="date_id", how="left")
    monthly = (
        fact_with_date2
        .groupby(["year", "month", "month_name"])
        .agg(
            revenue     = ("revenue_usd",  "sum"),
            spend       = ("ad_spend_usd", "sum"),
            conversions = ("conversions",  "sum"),
            impressions = ("impressions",  "sum"),
            clicks      = ("clicks",       "sum"),
        )
        .reset_index()
        .sort_values(["year", "month"])
    )
    monthly["ROAS"] = (monthly["revenue"] / monthly["spend"]).round(2)
    monthly["CPA"]  = (monthly["spend"]   / monthly["conversions"]).round(2)

    # ── KPIs globales ──────────────────────────────
    totals = {
        "total_revenue":      round(float(fact["revenue_usd"].sum()), 2),
        "total_spend":        round(float(fact["ad_spend_usd"].sum()), 2),
        "total_conversions":  int(fact["conversions"].sum()),
        "total_impressions":  int(fact["impressions"].sum()),
        "total_clicks":       int(fact["clicks"].sum()),
        "overall_ROAS":       round(float(fact["revenue_usd"].sum() / fact["ad_spend_usd"].sum()), 2),
        "overall_CPA":        round(float(fact["ad_spend_usd"].sum() / fact["conversions"].sum()), 2),
        "overall_CVR_pct":    round(float(fact["conversions"].sum() / fact["clicks"].sum() * 100), 2),
        "overall_CTR_pct":    round(float(fact["clicks"].sum() / fact["impressions"].sum() * 100), 2),
    }

    metrics = {
        "kpis_globales":        totals,
        "metricas_por_cliente": client_agg,
        "metricas_por_canal":   channel_agg,
        "evolucion_mensual":    monthly,
        "costo_marginal":       weekly,
        "fact_enriquecida":     fact,
    }

    print(f"   ✅ KPIs globales calculados")
    print(f"   ✅ ROAS dinámico por cliente")
    print(f"   ✅ CPA ajustado por canal")
    print(f"   ✅ LTV estimado por cliente")
    print(f"   ✅ Costo marginal semanal\n")
    return metrics


# ─────────────────────────────────────────────
# 5. REPORTE EN CONSOLA
# ─────────────────────────────────────────────

def print_report(metrics: dict) -> None:
    """Imprime un resumen ejecutivo en consola."""
    sep = "─" * 65

    print(sep)
    print("  EDIBS SCHOOL — REPORTE ANALÍTICO Q1 2024")
    print(sep)

    t = metrics["kpis_globales"]
    print(f"\n  📊 KPIs GLOBALES")
    print(f"     Revenue total   : ${t['total_revenue']:>12,.2f}")
    print(f"     Ad Spend total  : ${t['total_spend']:>12,.2f}")
    print(f"     ROAS global     : {t['overall_ROAS']:>12.2f}x")
    print(f"     CPA promedio    : ${t['overall_CPA']:>12.2f}")
    print(f"     CVR             : {t['overall_CVR_pct']:>12.2f} %")
    print(f"     CTR             : {t['overall_CTR_pct']:>12.2f} %")
    print(f"     Conversiones    : {t['total_conversions']:>12,}")
    print(f"     Impresiones     : {t['total_impressions']:>12,}")

    print(f"\n  👥 ROAS DINÁMICO POR CLIENTE")
    cols = ["client_name", "ROAS_dinamico", "CPA", "CVR_pct", "LTV_estimado"]
    print(metrics["metricas_por_cliente"][cols].to_string(index=False))

    print(f"\n  📡 CPA AJUSTADO POR CANAL")
    cols2 = ["channel_name", "ROAS", "CPA_neto", "CPA_ajustado", "CVR_pct"]
    print(metrics["metricas_por_canal"][cols2].to_string(index=False))

    print(f"\n  📅 EVOLUCIÓN MENSUAL")
    cols3 = ["month_name", "revenue", "spend", "ROAS", "CPA", "conversions"]
    print(metrics["evolucion_mensual"][cols3].to_string(index=False))

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────
# 6. EXPORTACIÓN DE RESULTADOS
# ─────────────────────────────────────────────

def export_results(metrics: dict, output_path: str = "edibs_resultados.xlsx") -> None:
    """
    Exporta todas las métricas a un archivo Excel con múltiples hojas.
    También guarda los KPIs globales como JSON.
    """
    print(f"💾 Exportando resultados a '{output_path}'...")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        metrics["metricas_por_cliente"].to_excel(writer, sheet_name="Clientes",       index=False)
        metrics["metricas_por_canal"].to_excel(  writer, sheet_name="Canales",        index=False)
        metrics["evolucion_mensual"].to_excel(   writer, sheet_name="Mensual",        index=False)
        metrics["costo_marginal"].to_excel(      writer, sheet_name="Costo_Marginal", index=False)
        metrics["fact_enriquecida"].to_excel(    writer, sheet_name="Fact_Table",     index=False)

    kpi_path = output_path.replace(".xlsx", "_kpis.json")
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(metrics["kpis_globales"], f, indent=2, ensure_ascii=False)

    print(f"   ✅ Excel  → {output_path}")
    print(f"   ✅ JSON   → {kpi_path}\n")


# ─────────────────────────────────────────────
# 7. ENTRYPOINT
# ─────────────────────────────────────────────

def main():
    print("\n" + "═" * 65)
    print("  EDIBS School · Pipeline Analítico")
    print("═" * 65 + "\n")

    # Paso 1 — Carga
    dfs = load_dataframes()

    # Paso 2 — Esquema estrella
    schema = build_star_schema(dfs)

    # Paso 3 — Métricas
    metrics = compute_metrics(schema)

    # Paso 4 — Reporte
    print_report(metrics)

    # Paso 5 — Exportación
    export_results(metrics, output_path="edibs_resultados.xlsx")

    print("✅ Pipeline completado con éxito.\n")


if __name__ == "__main__":
    main()
