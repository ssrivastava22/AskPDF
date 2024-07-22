from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import cassio
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

os.environ['OPENAI_API_KEY'] = '<enter api key>'
os.environ['ASTRA_DB_ID'] = '62db5132-7ca6-4bd4-8f29-a7f7bf3abc33'
os.environ['ASTRA_DB_APP_TOKEN'] = 'AstraCS:nusoefSiysYweSRTbWQyCRhM:b3c5a2745a8194aaae6dfb63126aa9f9d4d13ee0d9a5799e0e0b93d0f1626ff7'

cassio.init(token=os.environ['ASTRA_DB_APP_TOKEN'], database_id=os.environ['ASTRA_DB_ID'])
model = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])
embedding_model = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_pdf(filepath):
    pdf = PdfReader(filepath)
    raw = ''
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            raw += text
    return raw

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            raw_text = process_pdf(filepath)

            splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=800,
                chunk_overlap=200,
                length_function=len
            )
            chunks = splitter.split_text(raw_text)

            vectorstore = Cassandra(
                embedding=embedding_model,
                table_name="pdf_qa",
                session=None,
                keyspace=None
            )

            vectorstore.add_texts(chunks[:100])
            vectorstoreindex = VectorStoreIndexWrapper(vectorstore=vectorstore)

            query = request.form.get('query')
            if query:
                answer = vectorstoreindex.query(query, llm=model)
                documents = vectorstore.similarity_search_with_score(query, k=4)
                return render_template('index.html', query=query, answer=answer, documents=documents)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
