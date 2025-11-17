import subprocess
import psycopg2

def get_trip_summary_from_db(name):
    try:
        conn = psycopg2.connect("postgresql://localhost")
        cur = conn.cursor()
        cur.execute("SELECT name, destination, start_date, end_date, budget FROM trip_requests WHERE name = %s", (name,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return "No trips found for that user."

        summaries = []
        for r in rows:
            summaries.append(f"{r[0]} has a trip to {r[1]} from {r[2]} to {r[3]} with a budget of {r[4]}")
        return "\n".join(summaries)

    except Exception as e:
        return f"Database error: {e}"

def get_llm_response(prompt):
    try:
        # Run ollama CLI with echo piped to stdin, no --prompt flag
        process = subprocess.Popen(
            ['ollama', 'run', 'llama3'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(prompt)

        if process.returncode != 0:
            return f"Error calling Ollama: {stderr.strip()}"
        return stdout.strip()

    except Exception as e:
        return f"Error running Ollama CLI: {e}"

def main():
    user_name = input("Enter user name: ")
    db_summary = get_trip_summary_from_db(user_name)
    if "No trips found" in db_summary or "Database error" in db_summary:
        print(db_summary)
        return

    prompt = f"Summarize this trip info:\n{db_summary}\n\nProvide a short, friendly summary."

    llm_output = get_llm_response(prompt)
    print("Trip summary:")
    print(llm_output)

if __name__ == "__main__":
    main()
