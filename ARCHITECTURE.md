```mermaid
flowchart LR
    A["Sample docs"] --> B["Qwen: what's wrong?"]
    B --> C["Pick template + params"]
    C --> D["Filter 5M docs"]
    D --> E["Train 5 min"]
    E --> F{"BPB improved?"}
    F -->|yes| G["Keep rule"]
    F -->|no| H["Revert"]
    G & H -.-> A
```
