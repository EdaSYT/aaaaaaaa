import streamlit as st
import pandas as pd
import plotly.express as px
from gurobipy import Model, GRB, quicksum

# =========================================================
# STREAMLIT AYARLARI
# =========================================================
st.set_page_config(
    page_title="Montaj Hattı Optimizasyonu",
    layout="wide"
)

st.title("Montaj Hattı İşçi Atama Optimizasyonu")

# =========================================================
# VERİ
# =========================================================
I = range(1, 64)
J = range(1, 37)
W = range(1, 37)

t = {
    1: 2.43, 2: 9.79, 3: 2.12, 4: 9.92, 5: 4.66, 6: 11.58,
    7: 1.01, 8: 1.44, 9: 9.66, 10: 10.30, 11: 0.49, 12: 7.13,
    13: 7.18, 14: 2.44, 15: 3.58, 16: 4.90, 17: 3.21, 18: 7.78,
    19: 11.27, 20: 11.35, 21: 0.80, 22: 3.31, 23: 9.83, 24: 0.80,
    25: 4.61, 26: 5.20, 27: 11.89, 28: 6.30, 29: 13.32, 30: 0.98,
    31: 14.20, 32: 6.13, 33: 0.98, 34: 14.49, 35: 3.14, 36: 12.12,
    37: 1.07, 38: 5.14, 39: 5.63, 40: 0.57, 41: 10.13, 42: 0.90,
    43: 1.39, 44: 1.43, 45: 0.51, 46: 10.74, 47: 5.65, 48: 7.38,
    49: 1.71, 50: 15.09, 51: 7.31, 52: 6.93, 53: 10.72, 54: 1.31,
    55: 6.45, 56: 2.39, 57: 0.89, 58: 11.06, 59: 8.02, 60: 6.48,
    61: 3.13, 62: 0.53, 63: 7.74
}

P = [(i, i + 1) for i in range(1, 63)]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Model Parametreleri")

U_MAX = st.sidebar.slider(
    "Maksimum Operatör Doluluk (%)",
    50,
    100,
    95
) / 100

L = st.sidebar.slider(
    "Maksimum Mesafe",
    1,
    20,
    4
)

D = st.sidebar.number_input(
    "Hedef Üretim",
    min_value=1,
    value=32
)

T = st.sidebar.number_input(
    "Vardiya Süresi",
    min_value=1,
    value=510
)

worker_count = st.sidebar.slider(
    "Operatör Sayısı",
    1,
    36,
    29
)

solve_button = st.sidebar.button("Modeli Çöz")

# =========================================================
# MESAFE MATRİSİ
# =========================================================
d = {
    j: {k: 2 * abs(j - k) for k in J}
    for j in J
}

BIG_M = sum(t.values())

# =========================================================
# MODEL
# =========================================================
def solve_model():

    m = Model("assembly_line")
    m.setParam("OutputFlag", 0)

    # Karar değişkenleri
    x = m.addVars(I, J, vtype=GRB.BINARY, name="x")
    y = m.addVars(W, J, vtype=GRB.BINARY, name="y")
    z = m.addVars(W, vtype=GRB.BINARY, name="z")

    l = m.addVars(J, lb=0, vtype=GRB.CONTINUOUS, name="l")
    q = m.addVars(W, J, lb=0, vtype=GRB.CONTINUOUS, name="q")
    U = m.addVars(W, lb=0, vtype=GRB.CONTINUOUS, name="U")

    C = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C")

    # =====================================================
    # 1) Her operasyon bir istasyona atanır
    # =====================================================
    for i in I:
        m.addConstr(
            quicksum(x[i, j] for j in J) == 1
        )

    # =====================================================
    # 2) Öncelik ilişkileri
    # =====================================================
    for i, h in P:
        m.addConstr(
            quicksum(j * x[i, j] for j in J)
            <=
            quicksum(j * x[h, j] for j in J)
        )

    # =====================================================
    # 3) İstasyon yükü
    # =====================================================
    for j in J:
        m.addConstr(
            l[j] == quicksum(t[i] * x[i, j] for i in I)
        )

    # =====================================================
    # 4) Her istasyona 1 operatör
    # =====================================================
    for j in J:
        m.addConstr(
            quicksum(y[w, j] for w in W) == 1
        )

    # =====================================================
    # 5) Operatör kullanım bağlantısı
    # =====================================================
    for w in W:
        for j in J:
            m.addConstr(
                y[w, j] <= z[w]
            )

    # =====================================================
    # 6) q tanımı
    # =====================================================
    for w in W:
        for j in J:

            m.addConstr(q[w, j] <= l[j])

            m.addConstr(
                q[w, j] <= BIG_M * y[w, j]
            )

            m.addConstr(
                q[w, j] >= l[j] - BIG_M * (1 - y[w, j])
            )

    # =====================================================
    # 7) Operatör yükü <= çevrim süresi
    # =====================================================
    for w in W:
        m.addConstr(
            quicksum(q[w, j] for j in J) <= C
        )

    # =====================================================
    # 8) İstasyon yükü <= çevrim süresi
    # =====================================================
    for j in J:
        m.addConstr(
            l[j] <= C
        )

    # =====================================================
    # 9) Operatör doluluk oranı
    # =====================================================
    for w in W:

        m.addConstr(
            U[w] == (D / T) * quicksum(q[w, j] for j in J)
        )

        # KRİTİK KISIT
        m.addConstr(
            U[w] <= U_MAX
        )

    # =====================================================
    # 10) Mesafe kısıtı
    # =====================================================
    for w in W:
        for j in J:
            for k in J:

                if j < k and d[j][k] > L:

                    m.addConstr(
                        y[w, j] + y[w, k] <= 1
                    )

    # =====================================================
    # 11) Operatör sayısı
    # =====================================================
    m.addConstr(
        quicksum(z[w] for w in W) == worker_count
    )

    # =====================================================
    # AMAÇ
    # =====================================================
    m.setObjective(C, GRB.MINIMIZE)

    # =====================================================
    # ÇÖZ
    # =====================================================
    m.optimize()

    return m, x, y, z, l, q, U, C

# =========================================================
# ÇÖZÜM
# =========================================================
if solve_button:

    with st.spinner("Model çözülüyor..."):

        m, x, y, z, l, q, U, C = solve_model()

    # =====================================================
    # FEASIBLE KONTROL
    # =====================================================
    if m.status == GRB.INFEASIBLE:

        st.error("Model uygun çözüm bulamadı.")

        m.computeIIS()
        m.write("infeasible.ilp")

        st.warning(
            "IIS dosyası oluşturuldu: infeasible.ilp"
        )

    elif m.status == GRB.OPTIMAL:

        st.success("Optimal çözüm bulundu.")

        # =================================================
        # KPI
        # =================================================
        used_workers = sum(
            int(round(z[w].X))
            for w in W
        )

        reachable_output = T / C.X

        max_util = max(
            100 * U[w].X
            for w in W
        )

        avg_util = sum(
            100 * U[w].X
            for w in W if z[w].X > 0.5
        ) / used_workers

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Çevrim Süresi",
            f"{C.X:.2f}"
        )

        col2.metric(
            "Operatör Sayısı",
            used_workers
        )

        col3.metric(
            "Üretim Kapasitesi",
            f"{reachable_output:.2f}"
        )

        col4.metric(
            "Maksimum Doluluk",
            f"%{max_util:.2f}"
        )

        # =================================================
        # İSTASYON TABLOSU
        # =================================================
        station_data = []

        for j in J:

            ops = []

            for i in I:
                if x[i, j].X > 0.5:
                    ops.append(i)

            assigned_worker = None

            for w in W:
                if y[w, j].X > 0.5:
                    assigned_worker = w

            station_data.append({
                "İstasyon": j,
                "Operasyonlar": str(ops),
                "Operatör": assigned_worker,
                "Yük": round(l[j].X, 2)
            })

        df_station = pd.DataFrame(station_data)

        st.subheader("İstasyon Atamaları")

        st.dataframe(
            df_station,
            use_container_width=True
        )

        # =================================================
        # OPERATÖR TABLOSU
        # =================================================
        worker_data = []

        for w in W:

            if z[w].X > 0.5:

                stations = []

                for j in J:
                    if y[w, j].X > 0.5:
                        stations.append(j)

                load = sum(
                    q[w, j].X
                    for j in J
                )

                worker_data.append({
                    "Operatör": w,
                    "İstasyonlar": str(stations),
                    "Ürün Başı Yük": round(load, 2),
                    "Vardiya Yükü": round(D * load, 2),
                    "Doluluk (%)": round(100 * U[w].X, 2)
                })

        df_worker = pd.DataFrame(worker_data)

        st.subheader("Operatör Bilgileri")

        st.dataframe(
            df_worker,
            use_container_width=True
        )

        # =================================================
        # GRAFİK
        # =================================================
        fig = px.bar(
            df_worker,
            x="Operatör",
            y="Doluluk (%)",
            title="Operatör Doluluk Oranları"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )