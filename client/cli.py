import argparse
import requests
import time
import os
import json

def upload_file(file_path, ocr_cache, prompt, prompt_file=None, model='llama3.2-vision', strategy='marker'):
    ocr_url = os.getenv('OCR_URL', 'http://localhost:8000/ocr')
    files = {'file': open(file_path, 'rb')}
    data = {'ocr_cache': ocr_cache, 'model': model, 'strategy': strategy}

    try:
        if prompt_file:
            prompt = open(prompt_file, 'r').read()
    except FileNotFoundError:
        print(f"Prompt file not found: {prompt_file}")
        return None

    if prompt:
        data['prompt'] = prompt

    response = requests.post(ocr_url, files=files, data=data)
    if response.status_code == 200:
        respObject = response.json()
        if respObject.get('task_id'):
            return {
                "task_id": respObject.get('task_id')
            }
        else:
            return {
                "text": respObject.get('text')  # Sync mode support
            }
    else:
        print(f"Failed to upload file: {response.text}")
        return None

def get_result(task_id, print_progress=False):
    extracted_text_printed_once = False
    result_url = os.getenv('RESULT_URL', f'http://localhost:8000/ocr/result/{task_id}')
    while True:
        response = requests.get(result_url)
        result = response.json()
        if result['state'] != 'SUCCESS' and print_progress:
            task_info = result.get('info')
            if task_info is not None:
                if task_info.get('extracted_text'):
                    if not extracted_text_printed_once:
                        extracted_text_printed_once = True
                        print("Extracted text:\n" + task_info.get('extracted_text'))
                    else:
                        del task_info['extracted_text']
                del task_info['start_time']
            print(result)
        if response.status_code == 200:
            if result['state'] == 'SUCCESS':
                return result['result']
            elif result['state'] == 'FAILURE':
                print("OCR task failed.")
                return None
        time.sleep(2)  # Wait for 2 seconds before checking again

def parse_extracted_data(extracted_text):
    """
    Parses the LLM's output and constructs the JSON object.
    """
    data = {
        "invoice": {
            "numeroFatura": None,
            "dataEmissao": None,
            "dataVencimento": None,
            "tipoDocumento": None,
            "estadoDocumento": None,
            "atcud": None,
            "certificado": None,
            "totalImpostos": None,
            "totalComImpostos": None,
            "identificacaoUnica": None,
            "hash": None,
            "qrCodeFields": {
                "A_Issuers_Tax_ID": None,
                "B_Buyers_Tax_ID": None,
                "C_Buyers_Country": None,
                "D_Document_Type": None,
                "E_Document_Status": None,
                "F_Document_Date": None,
                "G_Document_Unique_ID": None,
                "H_ATCUD": None,
                "I1_Tax_Region": None,
                "I2_Taxable_Amount_Exempt_VAT": None,
                "I3_Taxable_Amount_Reduced_VAT": None,
                "I4_VAT_Amount_Reduced_Rate": None,
                "I5_Taxable_Amount_Intermediate_VAT": None,
                "I6_VAT_Amount_Intermediate_Rate": None,
                "I7_Taxable_Amount_Standard_VAT": None,
                "I8_VAT_Amount_Standard_Rate": None,
                "N_Total_Taxes": None,
                "O_Total_With_Taxes": None,
                "Q_Hash_Characters": None,
                "R_Certificate_Number": None
            }
        },
        "emitente": {
            "taxID": None,
            "nome": None,
            "endereco": None,
            "codigoPostal": None,
            "pais": None,
            "telefone": None,
            "email": None
        },
        "cliente": {
            "taxID": None,
            "nome": None,
            "endereco": None,
            "codigoPostal": None,
            "pais": None,
            "telefone": None,
            "email": None
        },
        "itens": [],
        "subtotal": None
    }

    # Split the text into lines
    lines = extracted_text.split('\n')
    for line in lines:
        line = line.strip()
        # Parse invoice fields
        if line.startswith('Invoice Number:'):
            data['invoice']['numeroFatura'] = line.split(':', 1)[1].strip()
        elif line.startswith('Issue Date:'):
            data['invoice']['dataEmissao'] = line.split(':', 1)[1].strip()
        elif line.startswith('Due Date:'):
            data['invoice']['dataVencimento'] = line.split(':', 1)[1].strip()
        # Continue parsing other invoice fields...

        # Parse emitente fields
        elif line.startswith('Issuer Tax ID:'):
            data['emitente']['taxID'] = line.split(':', 1)[1].strip()
        elif line.startswith('Issuer Name:'):
            data['emitente']['nome'] = line.split(':', 1)[1].strip()
        # Continue parsing other emitente fields...

        # Parse cliente fields
        elif line.startswith('Buyer Tax ID:'):
            data['cliente']['taxID'] = line.split(':', 1)[1].strip()
        elif line.startswith('Buyer Name:'):
            data['cliente']['nome'] = line.split(':', 1)[1].strip()
        # Continue parsing other cliente fields...

        # Parse item lines
        elif line.startswith('Item:'):
            item_details = line.split(':', 1)[1].strip()
            item_parts = item_details.split(',')
            item = {
                "codigo": None,
                "descricao": None,
                "quantidade": None,
                "precoUnitario": None,
                "descontoValor": None,
                "descontoPercentagem": None,
                "iva": None,
                "total": None
            }
            for part in item_parts:
                part = part.strip()
                if part.startswith('Code'):
                    item['codigo'] = part.split('Code', 1)[1].strip()
                elif part.startswith('Description'):
                    item['descricao'] = part.split('Description', 1)[1].strip()
                elif part.startswith('Quantity'):
                    item['quantidade'] = float(part.split('Quantity',1)[1].strip())
                elif part.startswith('Unit Price'):
                    item['precoUnitario'] = float(part.split('Unit Price',1)[1].strip())
                # Continue parsing other item fields...
            data['itens'].append(item)

        # Parse subtotal
        elif line.startswith('Subtotal:'):
            data['subtotal'] = float(line.split(':', 1)[1].strip())

    return data

def clear_cache():
    clear_cache_url = os.getenv('CLEAR_CACHE_URL', 'http://localhost:8000/ocr/clear_cache')
    response = requests.post(clear_cache_url)
    if response.status_code == 200:
        print("OCR cache cleared successfully.")
    else:
        print(f"Failed to clear OCR cache: {response.text}")

def llm_pull(model='llama3.2-vision'):
    ollama_pull_url = os.getenv('LLM_PULL_API_URL', 'http://localhost:8000/llm_pull')
    response = requests.post(ollama_pull_url, json={"model": model})
    if response.status_code == 200:
        print("Model pulled successfully.")
    else:
        print(f"Failed to pull the model: {response.text}")

def llm_generate(prompt, model='llama3.2-vision'):
    ollama_url = os.getenv('LLM_GENERATE_API_URL', 'http://localhost:8000/llm_generate')
    response = requests.post(ollama_url, json={"model": model, "prompt": prompt})
    if response.status_code == 200:
        print(response.json().get('generated_text'))
    else:
        print(f"Failed to generate text: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="CLI for OCR and Ollama operations.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Sub-command for uploading a file
    ocr_parser = subparsers.add_parser('ocr', help='Upload a file to the OCR endpoint and get the result.')
    ocr_parser.add_argument('--file', type=str, default='examples/invoice.pdf', help='Path to the file to upload')
    ocr_parser.add_argument('--ocr_cache', default=True, action='store_true', help='Enable OCR result caching')
    ocr_parser.add_argument('--prompt', type=str, default=None, help='Prompt used for the Ollama model to fix or transform the file')
    ocr_parser.add_argument('--prompt_file', default=None, type=str, help='Prompt file name used for the Ollama model to fix or transform the file')
    ocr_parser.add_argument('--model', type=str, default='llama3.2-vision', help='Model to use for the Ollama endpoint')
    ocr_parser.add_argument('--strategy', type=str, default='marker', help='OCR strategy to use for the file')
    ocr_parser.add_argument('--print_progress', default=True, action='store_true', help='Print the progress of the OCR task')

    # Sub-command for getting the result
    result_parser = subparsers.add_parser('result', help='Get the OCR result by specified task id.')
    result_parser.add_argument('--task_id', type=str, help='Task Id returned by the upload command')
    result_parser.add_argument('--print_progress', default=True, action='store_true', help='Print the progress of the OCR task')

    # Sub-command for clearing the cache
    clear_cache_parser = subparsers.add_parser('clear_cache', help='Clear the OCR result cache')

    # Sub-command for running Ollama
    ollama_parser = subparsers.add_parser('llm_generate', help='Run the Ollama endpoint')
    ollama_parser.add_argument('--prompt', type=str, required=True, help='Prompt for the Ollama model')
    ollama_parser.add_argument('--model', type=str, default='llama3.2-vision', help='Model to use for the Ollama endpoint')

    # Sub-command for pulling the Llama model
    ollama_pull_parser = subparsers.add_parser('llm_pull', help='Pull the latest Llama model from the Ollama API')
    ollama_pull_parser.add_argument('--model', type=str, default='llama3.2-vision', help='Model to pull from the Ollama API')

    args = parser.parse_args()

    if args.command == 'ocr':
        result = upload_file(args.file, args.ocr_cache, args.prompt, args.prompt_file, args.model)
        if result is None:
            print("Error uploading file.")
            return
        if result.get('text'):
            # LLM output received directly
            extracted_text = result.get('text')
            parsed_data = parse_extracted_data(extracted_text)
            print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
        elif result:
            print("File uploaded successfully. Task Id: " + result.get('task_id') + " Waiting for the result...")
            text_result = get_result(result.get('task_id'), args.print_progress)
            if text_result:
                # Assuming text_result is the LLM's output
                parsed_data = parse_extracted_data(text_result)
                print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
    elif args.command == 'result':
        text_result = get_result(args.task_id, args.print_progress)
        if text_result:
            parsed_data = parse_extracted_data(text_result)
            print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
    elif args.command == 'clear_cache':
        clear_cache()
    elif args.command == 'llm_generate':
        llm_generate(args.prompt, args.model)
    elif args.command == 'llm_pull':
        llm_pull(args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
