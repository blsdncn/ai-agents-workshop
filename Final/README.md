# Final Project for AI-Agents Workshop -- Vibecheck

- This is a project that acts as a knowledge gate that tied into the agentic loop of Claude Code
  - UI spawned upon claude code launch, hooks into tools that mutate code, normalizes the output of the `PreToolUse` hook into a code-change proposal, and compares it against an internal representation of the user's 'competence' to decide whether to block the changes.
  - Upon being blocked, Vibecheck quizzes you on the proposed changes, if you get it correct, it will update the competence model to make it less likely to block on similar questions in the future. If you get in incorrect, it records your misunderstanding in the competence model to continue to challenge you on similar changes.

## Demo Video

[![YouTube Video Thumbnail](https://img.youtube.com/vi/_tsv92gePac/hqdefault.jpg)](https://www.youtube.com/watch?v=_tsv92gePac)
