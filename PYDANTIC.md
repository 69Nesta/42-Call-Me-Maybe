Here’s a simple, clear example of how to use **Pydantic** in Python to:

1. Load JSON from a file
2. Validate and format it using a Pydantic model
3. Export the validated data back to a new JSON file

---

## ✅ Step 1: Install Pydantic

```bash
pip install pydantic
```

---

## ✅ Step 2: Example JSON File (`input.json`)

```json
{
  "id": 1,
  "name": "John Doe",
  "email": "john@example.com",
  "age": 30
}
```

---

## ✅ Step 3: Create a Pydantic Model and Process the File

### ✔ Example Using Pydantic v2 (current version)

```python
from pydantic import BaseModel, EmailStr
import json

# Define your data model
class User(BaseModel):
    id: int
    name: str
    email: EmailStr
    age: int

# Load JSON file
with open("input.json", "r") as f:
    data = json.load(f)

# Validate and parse data
user = User(**data)

# Export formatted JSON to new file
with open("output.json", "w") as f:
    f.write(user.model_dump_json(indent=4))

print("JSON validated and exported successfully!")
```

---

## ✅ If You Are Using Pydantic v1

Replace:

```python
user.model_dump_json(indent=4)
```

With:

```python
user.json(indent=4)
```

---

## ✅ Handling a List of Objects

If your JSON file contains a list:

```json
[
  {"id": 1, "name": "John", "email": "john@example.com", "age": 30},
  {"id": 2, "name": "Jane", "email": "jane@example.com", "age": 25}
]
```

Use:

```python
from typing import List

with open("input.json", "r") as f:
    data = json.load(f)

users = [User(**item) for item in data]

with open("output.json", "w") as f:
    json.dump([user.model_dump() for user in users], f, indent=4)
```

---

## 🔎 What Pydantic Does Here

* Validates types (`int`, `str`, `EmailStr`)
* Ensures required fields exist
* Automatically formats output JSON
* Raises clear validation errors if data is invalid

---

If you'd like, I can also show:

* Auto-sorting keys
* Removing null values
* Custom field renaming (aliases)
* Nested JSON models
* Converting snake_case ↔ camelCase

Just tell me what you need 👍
