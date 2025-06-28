# web server with one get api having single query parameter
from flask import Flask, request, jsonify
from clickup_search_LM_studio_with_summary import search_task
app = Flask(__name__)

@app.route('/api', methods=['GET'])
def get_data():
    # Get the query parameter
    query = request.args.get('query')
    
    if query is None:
        return jsonify({'error': 'Missing required parameter'}), 400
    
    # write a code to call the clickup_search_LM_studio_with_summary.py callGenAI() method with the parameter
    
    query_response = search_task(query)


    # Process the parameter and return response
    response = {
        'message': query_response,
        'status': 'success'
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
