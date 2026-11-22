> **Document ID:** `05` | **Phase:** Phase 1E — Multi-Agent Workflow Architecture | **Role:** Workflow & Governance  
> **Source Word Document:** `Multi_Agent_Workflow_Inspection_and_Recommendation.docx` | **Repository Path:** `Speech-Ench-Answers/05_Multi_Agent_Workflow_Inspection_and_Recommendation.md`

---
# Multi-Agent Capabilities Inspection & Architecture Recommendation (Speech-Enhancement)

**Overview:** No project files were modified and no tasks were executed during this inspection. Below is the verified breakdown of the multi-agent capabilities natively available in your Antigravity 2.0 environment, followed by a practical 3-agent configuration tailored to Speech-Enhancement.

## Part 1: Inspection of Antigravity 2.0 Multi-Agent Capabilities

### 1. Can Multiple Specialized Agents Work on the Same Project?

- **Yes.** Antigravity 2.0 natively supports multi-agent workflows through four core tools exposed in this session:
  - `define_subagent:` Defines a custom, specialized agent type for the conversation with its own Name, Description, SystemPrompt, and granular tool permissions (EnableWriteTools, EnableMCPTools, EnableSubagentTools).
  - `invoke_subagent:` Launches an instance of either a built-in subagent (self, research, image-generator, firestore-rules-author) or a custom subagent defined via define_subagent, assigning it a specific Role, Prompt, Model (inherit, pro, flash, flash_lite), and Workspace mode (inherit, branch, share).
  - `send_message:` Enables asynchronous point-to-point communication between agents using their unique conversationId.
  - `manage_subagents:` Lists live subagents (list) or terminates specific/all subagents (kill, kill_all).
- Additionally, Antigravity 2.0 provides the `/teamwork` slash command in the chat UI, which triggers an autonomous multi-agent team workflow for larger projects.

### 2. Can Agents Work in Parallel?

- **Yes.** Every `invoke_subagent` call launches the subagent asynchronously in the background with its own independent conversation context (`conversationId`).
- Multiple subagents can be launched in the same turn and execute simultaneously. The parent agent does not need to poll; Antigravity reactive wakeup system automatically wakes the parent agent whenever a subagent finishes its task or sends a message via `send_message`.

### 3. Can Agents Have Separate Responsibilities and Instructions?

- **Yes, enforced at two levels:**
1. **Instructional Separation (SystemPrompt & Role):** Each custom subagent defined via `define_subagent` receives a dedicated SystemPrompt locking in its domain scope, coding standards, and reporting format, plus a per-invocation Role and Prompt.
2. **Capability / Tool Separation (EnableWriteTools):** Setting `EnableWriteTools: false` restricts a subagent strictly to read-only research and code inspection (read_file, search tools, and agent messaging)—it physically lacks the tools to create, edit, or delete files or run shell commands. Setting `EnableWriteTools: true` grants file creation/editing (write_file, edit_file) and command execution (run_command).

### 4. Can Agents Review Another Agent's Work Without Modifying It?

- **Yes, with hard technical enforcement.** You can guarantee read-only review in two ways:
  - **Built-in research Subagent:** Runs in a read-only sandbox with no write tools and no network access, purpose-built for code inspection and auditing.
  - **Custom QA Subagent with EnableWriteTools: false:** Defined via `define_subagent(..., EnableWriteTools=False)`. Because write and command tools are omitted from its toolset at the platform level, the QA / Research Engineer agent cannot modify files even by accident.

### 5. How Are File Conflicts and Shared Workspace Changes Handled?

- `invoke_subagent` provides a `Workspace` parameter with three modes:
1. **Workspace: "inherit" (Default):** The subagent operates directly on the same directory (`Speech-Enhancement`) as the parent agent. Conflict Behavior: There is NO automatic file locking in inherit mode. If two write-enabled agents edit the same file concurrently in inherit mode, they will overwrite each other changes or fail edit_file target matching.
2. **Workspace: "branch" / "share":** Creates an isolated workspace branched/shared from the repository (requires an initialized Git/Mercurial repository).
- **Important Local Constraint:** Because `Speech-Enhancement` is currently a plain directory on OneDrive Desktop (not an initialized .git repository), `Workspace: "inherit"` is the active mode. Therefore, to guarantee zero file conflicts, at most ONE agent should have write access at a time (or write access should be partitioned by disjoint directory/phase).

### 6. Can Agents Independently Interact with Google Colab?

- **No direct automated execution on your remote Google Colab runtime.** Agents run locally on your Windows desktop (`Speech-Enhancement`). They do not have a live kernel connection to execute cells directly inside a browser-hosted Google Colab VM.
- **How Colab Interaction Actually Works:**
1. Agents author, review, and save .py scripts, configs/config.yaml, and the .ipynb Colab notebook locally in Speech-Enhancement/.
2. You sync/upload those files to Google Drive / Google Colab and execute the GPU commands in Colab.
3. You paste the Colab console output (or download the resulting logs/CSVs into Speech-Enhancement/results/) back to the Project Manager agent, which routes the logs to the QA and Implementation agents for analysis and next steps.

### 7. What Coordination Mechanisms Are Available Between Agents?

- **Hub-and-Spoke Message Passing (send_message + Reactive Wakeup):** Each subagent is assigned a conversationId when invoked. Subagents report structured findings back to the Lead / Project Manager agent upon completion or when blocked. The Lead agent can send follow-up instructions to an existing subagent conversationId via send_message (preserving that subagent full memory context rather than spawning a new agent).
- **Shared Workspace Files & Transcripts:** Agents sharing Workspace: "inherit" can read files written by other agents in Speech-Enhancement immediately after a step completes.

## Part 2: Practical Recommendation for a 3-Agent Configuration

Because our workspace is a single local folder (`Workspace: "inherit"`) and GPU execution happens externally in Google Colab, running multiple write-enabled agents in parallel would introduce race conditions without speeding up Colab GPU execution. The most reliable, conflict-free architecture is a **Hub-and-Spoke Sequential Pipeline with Parallel Read-Only QA:**

                     `+--------------------------------------+`

                     `|   1. PROJECT MANAGER (Main Agent)    |`

                     `|  * Orchestrates phases & approvals   |`

                     `|  * Interfaces with User & Colab logs |`

                     `|  * Write Tools: Disabled by policy   |`

                     `+----------+----------------+----------+`

                                `|                |`

          `1. Dispatch task      |                | 2. Dispatch audit`

          `(Write-enabled)       v                v (Strictly Read-Only)`

       `+---------------------------------+    +---------------------------------+`

       `| 2. IMPLEMENTATION ENGINEER      |    | 3. QA / RESEARCH ENGINEER       |`

       `| * define_subagent:              |    | * define_subagent:              |`

       `|   EnableWriteTools = True       |    |   EnableWriteTools = False      |`

       `| * Sole writer of src/, scripts/ |    | * Audits code, math, STFT loss, |`

       `|   configs/, notebooks/          |    |   speaker splits & Colab logs   |`

       `+---------------------------------+    +---------------------------------+`

### Role 1: Project Manager Agent (The Primary / Lead Agent in Chat)

- **Implementation:** Served by the main conversation agent so you have a single point of contact, approval gate, and visibility.
- **Responsibilities:** (1) Maintains the master implementation roadmap and enforces phase gates (never advancing to the next step without QA sign-off and your approval). (2) Dispatches scoped implementation tasks to the Implementation Engineer subagent. (3) Dispatches verification and code-review tasks to the QA / Research Engineer subagent. (4) Translates outputs into Word documents (.docx) when requested and provides exact copy-paste Google Colab commands for you to run.

### Role 2: Implementation Engineer Agent (impl-engineer)

- **Configuration (define_subagent):** Name: "impl-engineer" | EnableWriteTools: True (The ONLY subagent permitted to create or edit project code in src/, scripts/, configs/, notebooks/) | EnableSubagentTools: False | Workspace: "inherit".
- **Responsibilities:** (1) Implements modular Python scripts (01_download_dataset.py through 07_evaluate_and_report.py), src/ modules, and the Colab notebook strictly according to the locked configs/config.yaml specification. (2) Fixes bugs or edge cases flagged by the QA / Research Engineer. (3) Never runs dataset downloads or training jobs locally.

### Role 3: QA / Research Engineer Agent (qa-research-engineer)

- **Configuration (define_subagent):** Name: "qa-research-engineer" | EnableWriteTools: False (Hardware/platform-enforced read-only mode; cannot modify or overwrite any file) | EnableMCPTools: False | EnableSubagentTools: False | Workspace: "inherit".
- **Responsibilities:** Performs static code audits on every file written by impl-engineer before we hand it to you for Colab execution, verifying mathematical and architectural invariants:
  - stft_sc_factor = 0.1 and stft_mag_factor = 0.1 applied once (total_loss = l1 + 0.1 * sc + 0.1 * mag).
  - torch.linalg.vector_norm(..., ord=2) used on flattened 3D spectrograms.
  - model.train() called before training forward/backward passes, and model.eval() + torch.no_grad() used for validation/test inference.
  - Strict speaker independence (tr, cv, tt disjoint) and zero test-set leakage.
  - Item-level license verification (10283/2791/license_text) and Demucs dns48 (never master64) usage.
  - Audits Google Colab logs and evaluation CSVs returned from your Colab runs to verify statistical validity.

## Part 3: Recommended Execution Protocol (Zero-Conflict Workflow)

For every implementation phase (starting with running the Step 1 smoke test and then building Step 2 dataset/preprocessing scripts):

1. **Step A (Implement):** Project Manager sends a scoped task to impl-engineer (EnableWriteTools=True). Only impl-engineer writes to disk.
2. **Step B (Audit):** Once impl-engineer finishes, Project Manager invokes qa-research-engineer (EnableWriteTools=False) to inspect the newly written/modified files against our checklist.
3. **Step C (Fix if Needed):** If qa-research-engineer finds any discrepancy, Project Manager messages impl-engineer (using send_message to its existing conversationId) to apply the exact fix, followed by a re-check from qa-research-engineer.
4. **Step D (User / Colab Gate):** Project Manager presents the verified result and exact Colab commands to you and waits for your approval.
