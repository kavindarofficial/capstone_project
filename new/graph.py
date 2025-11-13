from graphviz import Digraph

# Initialize ER diagram
er = Digraph("Crypto_Analysis_ERD", format="png")
er.attr(rankdir="LR", splines="ortho", bgcolor="white")
er.attr('node', shape='rectangle', style='filled', fillcolor='lightgoldenrod1', fontname='Helvetica', fontsize='11')

# --- Entities ---
er.node("AnalysisConfig", "1. Analysis Configuration\n(Primary Inputs)")
er.node("CryptoData", "2. Cryptocurrency Data\n(Historical + Enriched)")
er.node("AnalysisModules", "3. Analysis & Insight Modules\n(EDA, Risk, Signals, Regime)")
er.node("Modeling", "4. Predictive Modeling\n(Model, Results, Metrics)")
er.node("Comparison", "5. Comparative Analysis\n(Ranking, Correlation)")
er.node("Outputs", "6. Outputs\n(Reports, Exports)")

# --- Relationships ---
er.attr('edge', arrowsize='0.8', color='gray25', fontname='Helvetica', fontsize='10')

er.edge("AnalysisConfig", "CryptoData", label="fetches data for")
er.edge("CryptoData", "AnalysisModules", label="used by")
er.edge("CryptoData", "Modeling", label="trains models on")
er.edge("Modeling", "Outputs", label="produces metrics for")
er.edge("AnalysisModules", "Outputs", label="provides insights to")
er.edge("CryptoData", "Comparison", label="used for comparison")
er.edge("Comparison", "Outputs", label="summarized in report")

# Optional: circular highlight around data flow core
er.attr('node', shape='circle', style='dashed', color='gray70')
er.node("CoreFlow", "Data Flow")
er.edge("AnalysisConfig", "CoreFlow", style='dotted')
er.edge("CoreFlow", "Outputs", style='dotted')

# Save and render
er.render("Crypto_Analysis_ERD", cleanup=True)
print("✅ ER Diagram generated: Crypto_Analysis_ERD.png")
