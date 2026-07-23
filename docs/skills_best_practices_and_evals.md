# Guia de Boas Práticas e Avaliação de Skills (Evals)

> **Fonte & Inspiração:** Palestra de **Philipp Schmid** (Engenheiro da Google DeepMind) sobre a necessidade crítica de autoria enxuta e avaliação contínua ("evals") para skills em agentes codificadores de IA.

---

## 💡 1. O que são Skills e seus Tipos

Skills são mecanismos que estendem as capacidades de agentes de IA codificadores através de instruções estruturadas em Markdown. Elas operam sob o conceito de **Divulgação Progressiva** (*Progressive Disclosure*):

1. **Gatilho (Frontmatter):** O agente lê apenas o cabeçalho (`name` e `description`) no início do contexto para decidir se precisa carregar a skill.
2. **Instruções Principais (`SKILL.md`):** Carregadas no contexto apenas quando ativadas.
3. **Recursos Secundários (`scripts/`, `references/`):** Lidos sob demanda caso as instruções apontem para eles.

### Tipos Principais de Skills

| Tipo | Descrição | Ciclo de Vida | Exemplo |
| :--- | :--- | :--- | :--- |
| **Skills de Capacidade**<br>*(Capability Skills)* | Ensinam ao modelo algo que ele ainda não consegue fazer de forma consistente no estado atual da técnica. | **Temporária**: Deve ser aposentada assim que o modelo base evoluir. | "Criar um novo app React com Vite e SSR" |
| **Skills de Preferência**<br>*(Preference Skills)* | Codificam fluxos de trabalho, convenções de código ou regras de negócio específicas da sua empresa/projeto. | **Duradoura**: Modelos de fundação não têm como adivinhar o contexto interno. | "Seguir o padrão de commits da empresa e regras de linting internas" |

---

## 🛠️ 2. Boas Práticas de Autoria (*Skill Authoring*)

### 🎯 Foco na Descrição e Triggers
A `description` no frontmatter YAML é o elemento mais crítico. Ela dita **quando** a skill deve (e não deve) ser acionada.
- Inclua o **porquê** e o **como** de forma direta.
- Use gatilhos explícitos (ex: *"Use quando o usuário pedir para..."*).

### ✍️ Escreva Diretrizes, não Redações
- Use tom imperativo e ordens diretas em vez de explicações passivas ou históricas.
- Evite parágrafos longos; prefira listas objetivas e tabelas.

### 🧩 Defina Objetivos e Restrições, não Scripts Engessados
- Se um processo for 100% determinístico e engessado, prefira escrever um **script tradicional (Bash/Python)**.
- O papel da skill é fornecer **objetivos claros, limites e restrições** para que a IA possa navegar com flexibilidade.

### 🛑 Inclua "Testes Negativos" nas Instruções
- Diga explicitamente quando o modelo **NÃO** deve acionar ou aplicar a skill.
- Isso previne o chamado ***over-triggering*** (quando a IA ativa a skill em contextos irrelevantes).

### ⚡ Eficiência de Tokens e Remoção de "No-ops"
- **Tamanho Limite:** O arquivo `SKILL.md` principal deve ser mantido com **menos de 500 linhas**.
- **Remova "No-ops":** Elimine instruções vazias ou genéricas que não alteram o comportamento da IA (ex: *"escreva um código limpo e elegante"* ou *"seja cuidadoso"*). Elas consomem tokens e dinheiro sem gerar valor real.

---

## 🧪 3. Criação e Execução de Avaliações (*Skill Evals*)

Sem testes (*evals*), é impossível saber se uma falha do agente ocorre porque a skill é ruim ou porque a tarefa é complexa.

```mermaid
flowchart TD
    A[Prompt de Teste] --> B[Roda Agente SEM Skill]
    A --> C[Roda Agente COM Skill]
    B --> D[Valida Resultado em Assertions/Regex]
    C --> E[Valida Resultado em Assertions/Regex]
    D & E --> F[Teste de Ablação: Comparar Taxa de Sucesso]
    F --> G{Modelo Base Atingiu Paridade?}
    G -- Sim --> H[Aposentar Skill (Retire)]
    G -- Não --> I[Manter/Refinar Skill]
```

### 1. Comece Pequeno (Dataset Enxuto)
- Crie um conjunto de **10 a 20 prompts de teste** por skill.
- Divida entre:
  - **Happy Paths (Casos Positivos):** Prompts onde a skill DEVE ser acionada e ser bem-sucedida.
  - **Negative Cases (Casos Negativos):** Prompts semelhantes onde a skill NÃO DEVE ser acionada.

### 2. Mantenha os Testes Baratos (Evite "LLM-as-a-judge")
- Em vez de usar um LLM caro para julgar se a resposta foi boa, use **expressões regulares (Regex)** e **asserções Python**.
- Verifique pontos concretos:
  - O arquivo esperado foi criado no caminho correto?
  - A função `X` foi importada da biblioteca correta?
  - O comando CLI específico foi executado?

### 3. Isolamento e Repetição de Execuções
- Como agentes não são estritamente determinísticos, execute cada caso de teste entre **3 e 6 vezes**.
- Garanta que as execuções ocorram em **ambientes isolados** (containers ou diretórios limpos) para impedir contaminação de contexto entre testes.

### 4. Testes de Ablação (*Ablation Tests*) e Aposentadoria
- Sempre meça o desempenho do agente **COM** e **SEM** a skill ativada.
- **Regra de Ouro:** Se o modelo base atualizado atingir a mesma taxa de sucesso *sem* a skill que tinha *com* a skill, **aposente a skill**. Isso economiza manutenção e janela de contexto.

---

## ✅ 4. Checklist de Validação para Novas Skills

Antes de publicar ou mergear uma nova skill no repositório, verifique:

- [ ] A `description` no frontmatter possui gatilhos positivos e negativos claros?
- [ ] O `SKILL.md` tem menos de 500 linhas?
- [ ] Foram removidas instruções genéricas/no-ops ("escreva código limpo", etc.)?
- [ ] As instruções focam em **objetivos e restrições**, delegando passos rígidos a scripts?
- [ ] Há pelo menos 10 prompts de avaliação (positivos e negativos) definidos?
- [ ] Foi realizado um teste de ablação para confirmar que a skill realmente traz ganho de desempenho em relação ao modelo base?
