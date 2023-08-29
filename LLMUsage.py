import json
import os
import datetime

class LLMUsage:

    @staticmethod
    def add_usage_details(details):
        usage = {
            "Cost($)": details.total_cost,
            "Prompt Tokens": details.prompt_tokens,
            "Completion Tokens": details.completion_tokens,
            "Tokens Used": details.total_tokens
        }

        
        file_name = str(datetime.datetime.now().date()) + '_llm_usage.json'
        file_path = os.path.join('daily_llm_usage', file_name)

        usage_list = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                usage_list = json.load(json_file)

        with open(file_path, 'w') as json_file:
            usage_list.append(usage)
            usage_list = json.dump(usage_list, json_file)
