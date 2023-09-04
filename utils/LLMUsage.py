import json
import os
import datetime

class LLMUsage:

    @staticmethod
    def add_usage_details(details):
        curr_date = datetime.datetime.now()
        usage = {
            "Cost($)": details.total_cost,
            "Prompt Tokens": details.prompt_tokens,
            "Completion Tokens": details.completion_tokens,
            "Tokens Used": details.total_tokens,
            "Time": curr_date.strftime('%I:%M:%S %p')
        }

        
        file_name = str(curr_date.date()) + '_llm_usage.json'
        file_path = os.path.join('daily_llm_usage', file_name)

        usage_details = {
            "Total Cost($)": details.total_cost,
            "Details": [usage]
        }
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                existing_usage_details = json.load(json_file)
                usage_details["Total Cost($)"] += existing_usage_details["Total Cost($)"]
                usage_details["Details"] += existing_usage_details["Details"]

        with open(file_path, 'w') as json_file:
            json.dump(usage_details, json_file)
