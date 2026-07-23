"""Assemble the neutral gold set (43 Q) -> eval/questions_neutral.yaml.
Drops the budgeting collision. Verifies each expect path exists in the vault.
"""
import os
import yaml

VAULT = r"C:\Users\AlbertoDeCol\SecondBrain"

ITEMS = [
    # --- entities (18) ---
    {"q": "Chi è Simone Rizzo nel mondo dell'AI?", "expect": "archives/wiki/entities/simone-rizzo.md"},
    {"q": "Chi era Niklas Luhmann e per quale metodo di organizzazione delle note è conosciuto?", "expect": "archives/wiki/entities/niklas-luhmann.md"},
    {"q": "What is Qdrant and what is it used for?", "expect": "archives/wiki/entities/qdrant.md"},
    {"q": "A cosa serve Syncthing?", "expect": "archives/wiki/entities/syncthing.md"},
    {"q": "Chi è Tiago Forte?", "expect": "archives/wiki/entities/tiago-forte.md"},
    {"q": "Che cos'è Docker?", "expect": "archives/wiki/entities/docker.md"},
    {"q": "Chi è stato Dale Carnegie?", "expect": "archives/wiki/entities/dale-carnegie.md"},
    {"q": "What kind of company is Cerebras?", "expect": "archives/wiki/entities/cerebras.md"},
    {"q": "Who is KJ Rainey?", "expect": "archives/wiki/entities/kj-rainey.md"},
    {"q": "Quale libro ha scritto Antonio Gulli?", "expect": "archives/wiki/entities/antonio-gulli.md"},
    {"q": "Qual è il libro di Joe Friel presente nella vault in edizione italiana?", "expect": "archives/wiki/entities/joe-friel.md"},
    {"q": "Secondo la pagina su Seneca, quali sono i tre pilastri dello stoicismo classico?", "expect": "archives/wiki/entities/lucio-anneo-seneca.md"},
    {"q": "Which term did Andrej Karpathy coin?", "expect": "archives/wiki/entities/andrej-karpathy.md"},
    {"q": "Che scala di valutazione usa Steph Ango per ciò che ritiene valga la pena valutare?", "expect": "archives/wiki/entities/steph-ango.md"},
    {"q": "Quanti template offre n8n pronti all'uso?", "expect": "archives/wiki/entities/n8n.md"},
    {"q": "What is Mem0's default library stack?", "expect": "archives/wiki/entities/mem0.md"},
    {"q": "Quali sono le tre modalità desktop di Claude?", "expect": "archives/wiki/entities/claude-code.md"},
    {"q": "Chi è l'unico fondatore ad aver portato due aziende diverse nell'indice S&P 500?", "expect": "archives/wiki/entities/jack-dorsey.md"},
    # --- concepts (14, budgeting dropped) ---
    {"q": "Come funziona la Reciprocal Rank Fusion nell'hybrid retrieval e qual è la formula usata per combinare le liste?", "expect": "archives/wiki/concepts/hybrid-retrieval.md"},
    {"q": "Chi ha coniato la filosofia 'File Over App' e qual è la sua tesi centrale?", "expect": "archives/wiki/concepts/file-over-app.md"},
    {"q": "Qual è l'unità atomica di un knowledge graph e da quali tre elementi è composta?", "expect": "archives/wiki/concepts/knowledge-graph.md"},
    {"q": "Quali sono i tre concetti che il temporal knowledge graph adotta da Graphiti nella LLM Wiki?", "expect": "archives/wiki/concepts/temporal-knowledge-graph.md"},
    {"q": "Cosa si intende per finestra di tolleranza e cosa distingue iperarousal da ipoarousal?", "expect": "archives/wiki/concepts/window-of-tolerance.md"},
    {"q": "Quali sono le tre architetture di rete neurale fondamentali e per quale tipo di dati è progettata ciascuna?", "expect": "archives/wiki/concepts/neural-network-architectures.md"},
    {"q": "Qual è la differenza principale tra un container e una macchina virtuale?", "expect": "archives/wiki/concepts/containerization.md"},
    {"q": "What are the three layers of a Company Brain?", "expect": "archives/wiki/concepts/company-brain.md"},
    {"q": "Cos'è una nota atomica e da quale metodo di Niklas Luhmann deriva?", "expect": "archives/wiki/concepts/atomic-note.md"},
    {"q": "Nell'esempio di Cartabox sul know-how management, qual è il problema legato a Dante e la soluzione proposta?", "expect": "archives/wiki/concepts/know-how-management.md"},
    {"q": "What are the three steps of the Smart Loop in Smart Connections?", "expect": "archives/wiki/concepts/smart-connections.md"},
    {"q": "What problem does Learning-Driven Development (LDD) try to solve?", "expect": "archives/wiki/concepts/learning-driven-development.md"},
    {"q": "Which shell does the vault's SessionStart hook actually run through, and how was it verified?", "expect": "archives/wiki/concepts/sessionstart-hook-git-bash.md"},
    {"q": "Quali tre tipi di capacità espone un server MCP a un client LLM?", "expect": "archives/wiki/concepts/model-context-protocol.md"},
    # --- factual from sources/weekly (11, budgeting dropped) ---
    {"q": "Quando ho installato n8n in locale con Docker, su quale porta rispondeva l'editor nel browser?", "expect": "archives/wiki/sources/docker-n8n-local-install-session-2026-06-02.md"},
    {"q": "Qual è il comando da CLI per ispezionare la memoria completa che l'agente conserva su di me?", "expect": "archives/wiki/sources/hermes-agent-memory-system.md"},
    {"q": "Per quel progetto di trascrizione desktop che poi ho archiviato, quale modello avevo scelto come default per girare bene su CPU?", "expect": "archives/wiki/sources/echomind-project-retrospective.md"},
    {"q": "Chi era il docente indicato nel master sul diritto dell'ambiente che ho frequentato?", "expect": "archives/wiki/sources/master-diritto-ambientale-ascheri-academy.md"},
    {"q": "Nel blueprint di knowledge base di Cerebras, con quale soglia di IDF veniva attivato un burst di messaggi per renderlo ricercabile?", "expect": "archives/wiki/sources/cerebras-how-we-built-our-knowledge-base.md"},
    {"q": "Nel runbook del mio motore di retrieval locale, verso quale database deve puntare la test suite per non distruggere l'indice di produzione?", "expect": "archives/wiki/sources/sparcos-rag-build-runbook.md"},
    {"q": "In quella call di allineamento sui clienti manifatturieri, cosa aveva costruito Luca intorno alla documentazione di Power BI per dimostrare il valore di Claude e Obsidian?", "expect": "archives/wiki/sources/startup-allineamento-meeting-2026-06-04.md"},
    {"q": "Nel quickscan operativo dell'azienda cartaria, quale gap riguardava il personale commerciale che generava richieste sbagliate a monte?", "expect": "archives/wiki/sources/quickscan-cartotrentina-2026-06-05.md"},
    {"q": "In quella riunione in un'azienda manifatturiera italiana, entro quanto tempo ci si aspettava che un neoassunto diventasse operativo?", "expect": "archives/wiki/sources/fireflies-jun09-2026-manufacturing-meeting.md"},
    {"q": "Con quale strumento e in quanto tempo avevo costruito l'app di shopfloor management diventata standard aziendale per la Lean Production?", "expect": "archives/wiki/sources/naitive-hfarm-phase1-admission.md"},
    {"q": "Quella settimana in cui Syncthing era andato fuori sync, qual era stata la causa diagnosticata e su quale macchina?", "expect": "archives/weekly/2026-W29.md"},
]


def main():
    missing = [it for it in ITEMS if not os.path.isfile(os.path.join(VAULT, it["expect"]))]
    if missing:
        print("MISSING PATHS:")
        for m in missing:
            print("  ", m["expect"])
        raise SystemExit("Aborting: fix paths first.")
    with open("eval/questions_neutral.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(ITEMS, f, allow_unicode=True, sort_keys=False, width=1000)
    print(f"wrote eval/questions_neutral.yaml with {len(ITEMS)} questions (all paths verified)")


if __name__ == "__main__":
    main()
