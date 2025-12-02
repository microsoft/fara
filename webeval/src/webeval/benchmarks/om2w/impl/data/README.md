## Example

You can refer to the [example](https://github.com/OSU-NLP-Group/Online-Mind2Web/tree/main/data/example/fb7b4f784cfde003e2548fdf4e8d6b4f) format when conducting auto-eval. For each task, the result should be stored in a folder named as its `task_id`, containing :

- `trajectory/`: Stores screenshots of each step.
- `result.json`: Task metadata and action history.


## 🧠 Human Labels

The file `human_label.json` provides human evaluation labels for each task across different agents. The meaning of labels represent as following:

- `0`: ❌ **Failure** – The agent did **not** successfully complete the task.
- `1`: ✅ **Success** – The agent **successfully** completed the task.
- `2`: 🚫 **Not Executable** – The agent was **unable to execute** the task due to external or system-related limitations.

## ℹ️ Details on Label 2

The label 2 indicates that the agent could not execute the task at all. The specific reasons vary by agent:

- **SeeAct**: Some tasks raise internal bugs when the agent extracts elements from the web page.
- **Operator**: Certain websites (e.g., Reddit) are inaccessible, preventing task execution.
- **Claude Computer Use 3.7**: Due to updates on the websites, a few tasks became outdated during the testing phase..
- **Agent-E** and **Browser Use**: Label 2 typically corresponds to tasks blocked by **CAPTCHAs**. There are only **two** tasks for Agent-E and **one** task for Browser Use.

## Note
The results on the leaderboard are averaged over three runs, so there may be slight differences compared to the auto-eval results shown here.
