from flask import Flask, render_template, request, jsonify
import llm

# TODO: forward console messages to web

app = Flask(__name__)

# Sample conversation data
conversation = []

sys_prompt = "You are Bot. User will ask you questions. Please do your best to respond to the user helpfully. Keep your response inside one line, following \"Bot: \"\n\n"
myLLM = llm.largeLanguageModel('models/gpt2_torch_conversational_openwebtext-10k', 'cuda', sys_prompt)

@app.route('/', methods=['GET'])
def index():
    print("Rendering index.html")
    myLLM = llm.largeLanguageModel('models/gpt2_torch_conversational_openwebtext-10k', 'cuda', sys_prompt)
    return render_template('index.html', conversation=conversation)

@app.route('/process_message', methods=['POST'])
def process_message():
    print(request.form)
    user_message = request.json.get('user_message')
    conversation.append({'user': user_message})

    # You can replace this section with code to send the user input to your backend
    # and receive a response
    bot_response = myLLM.generate(user_message)

    conversation.append({'bot': bot_response.replace("Bot: ", "")})
    return jsonify({'bot_response': bot_response.replace("Bot: ", "")})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    data = request.json  # Extract JSON data from the request
    max_tokens = data.get('max_tokens')
    temperature = data.get('temperature')
    top_k = data.get('top_k')
    top_p = data.get('top_p')
    num_beams = data.get('num_beams')
    length_penalty = data.get('length_penalty')
    no_repeat_ngram_size = data.get('no_repeat_ngram_size')
    early_stopping = data.get('early_stopping')
    num_return_seqs = data.get('num_return_seqs')

    myLLM.set_max_length(max_tokens)
    myLLM.set_temp(temperature)
    myLLM.set_top_k(top_k)
    myLLM.set_top_p(top_p)
    myLLM.set_num_beams(num_beams)
    myLLM.set_length_penalty(length_penalty)
    myLLM.set_no_repeat_ngram_size(no_repeat_ngram_size)
    myLLM.set_early_stopping(early_stopping)
    myLLM.set_num_return_sequences(num_return_seqs)

    # Return a response TODO: Check input validity
    response_data = {
        'message': "Parameters updated successfully.",
        'success': True
    }

    # Return the JSON response
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
