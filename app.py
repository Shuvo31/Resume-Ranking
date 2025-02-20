from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import uvicorn
import fitz  # PyMuPDF for PDF extraction
import docx
from openai import OpenAI
import pandas as pd
from io import BytesIO

app = FastAPI(title="Resume Ranking", description="API to extract ranking criteria and score resumes.", version="1.0")

# Initialize OpenAI client
client = OpenAI(api_key = 'your-api-key')

def extract_text_from_pdf(file: UploadFile) -> str:
    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    return text

def extract_text_from_docx(file: UploadFile) -> str:
    doc = docx.Document(file.file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_criteria_from_text(text: str) -> List[str]:
    prompt = f"""
    Extract key hiring criteria from the following job description. Provide them as a list:
    {text}
    """
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip().split("\n")

@app.post("/extract-criteria")
async def extract_criteria(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith(".docx"):
            text = extract_text_from_docx(file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF or DOCX.")
        
        criteria = extract_criteria_from_text(text)
        return {"criteria": criteria}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def score_resume(text: str, criteria: List[str]) -> dict:
    scores = {}
    for crit in criteria:
        scores[crit] = sum(1 for word in crit.split() if word.lower() in text.lower())
    return scores

@app.post("/score-resumes")
async def score_resumes(criteria: List[str] = Form(...), files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif file.filename.endswith(".docx"):
                text = extract_text_from_docx(file)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}")
            
            scores = score_resume(text, criteria)
            results.append({"Candidate Name": file.filename, **scores, "Total Score": sum(scores.values())})
        
        df = pd.DataFrame(results)
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return {"message": "Scoring completed", "data": df.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
