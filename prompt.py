prompt = (
        "You are a support triage assistant.\n"
        "Return STRICT JSON with keys: category, priority, first_response_steps (array of strings).\n"
        f"Ticket: {example['text']}\n"
        "JSON:"
    )