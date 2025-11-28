# AI PDF Generator MVP - Complete Backend Implementation
# Requirements: pip install fastapi uvicorn langchain langchain-ollama langchain-core reportlab python-multipart

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import os
import json

# Initialize FastAPI
app = FastAPI(title="AI PDF Generator", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama (make sure Ollama is running locally with a model)
llm = OllamaLLM(model="llama3.1:8b",temperature=0)  # You can use mistral, llama2, codellama, etc.


# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str
    template: str


class GenerateResponse(BaseModel):
    success: bool
    filename: str
    message: str
    tokens_used: int = 0


# Template Definitions
TEMPLATES = {
    "business_report": {
        "name": "Business Report",
        "prompt_template": """You are a professional business analyst. Generate a detailed business report based on the following requirements:

{user_prompt}

Structure your response with these sections:
1. Executive Summary (2-3 paragraphs)
2. Detailed Analysis (3-4 paragraphs)
3. Key Findings (bullet points format)
4. Recommendations (2-3 paragraphs)

Use professional business language and provide specific, actionable insights.""",
        "style": "formal"
    },
    "invoice": {
        "name": "Invoice",
        "prompt_template": """Generate a detailed invoice based on these requirements:

{user_prompt}

Include:
- Company and client information
- Invoice number and date
- Itemized list of services/products with descriptions
- Individual prices and quantities
- Subtotal, tax (if applicable), and total amount
- Payment terms and methods

Format as a clear, professional invoice.""",
        "style": "structured"
    },
    "resume": {
        "name": "Resume/CV",
        "prompt_template": """Create a professional resume based on:

{user_prompt}

Include sections for:
- Professional Summary (2-3 sentences)
- Work Experience (with bullet points of achievements)
- Technical Skills
- Education
- Certifications (if applicable)

Use action verbs and quantify achievements where possible.""",
        "style": "professional"
    },
    "contract": {
        "name": "Contract",
        "prompt_template": """Draft a professional contract based on:

{user_prompt}

Include:
- Parties involved with full details
- Effective date and duration
- Scope of work/services
- Payment terms and schedule
- Terms and conditions
- Confidentiality clause
- Termination conditions
- Signature section

Use formal legal language appropriate for business contracts.""",
        "style": "legal"
    }
}


def create_pdf(content: str, template_name: str, filename: str):
    """Generate PDF from AI-generated content"""

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#1e3a8a',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#1e3a8a',
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
    )

    # Add title
    template_title = TEMPLATES[template_name]["name"]
    title = Paragraph(template_title, title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2 * inch))

    # Add generation date
    date_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    date_para = Paragraph(date_text, styles['Normal'])
    elements.append(date_para)
    elements.append(Spacer(1, 0.3 * inch))

    # Process content - split by sections
    sections = content.split('\n\n')

    for section in sections:
        if section.strip():
            # Check if it's a heading (starts with number or all caps)
            if (section.strip()[0].isdigit() or
                    section.strip().isupper() and len(section.strip()) < 50):
                para = Paragraph(section.strip(), heading_style)
            else:
                para = Paragraph(section.strip(), body_style)

            elements.append(para)
            elements.append(Spacer(1, 0.1 * inch))

    # Add footer
    elements.append(Spacer(1, 0.5 * inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor='gray',
        alignment=TA_CENTER
    )
    footer = Paragraph(
        "Generated by AI PDF Generator | Powered by LangChain + Ollama",
        footer_style
    )
    elements.append(footer)

    # Build PDF
    doc.build(elements)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI PDF Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "templates": "/api/templates",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Check if Ollama is running"""
    try:
        response = llm.invoke("test")
        return {"status": "healthy", "ollama": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "ollama": "disconnected", "error": str(e)}


@app.get("/api/templates")
async def get_templates():
    """Get available templates"""
    return {
        "templates": {
            key: {"name": val["name"], "style": val["style"]}
            for key, val in TEMPLATES.items()
        }
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_pdf(request: GenerateRequest):
    """Generate PDF from prompt using AI and template"""

    try:
        # Validate template
        if request.template not in TEMPLATES:
            raise HTTPException(status_code=400, detail="Invalid template")

        # Get template configuration
        template_config = TEMPLATES[request.template]

        # Create LangChain prompt (new LCEL syntax)
        prompt = PromptTemplate(
            input_variables=["user_prompt"],
            template=template_config["prompt_template"]
        )

        # Create chain using LCEL (LangChain Expression Language)
        chain = prompt | llm | StrOutputParser()

        # Generate content
        print(f"Generating content for: {request.prompt[:50]}...")
        ai_content = chain.invoke({"user_prompt": request.prompt})

        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/{request.template}_{timestamp}.pdf"

        # Create PDF
        create_pdf(ai_content, request.template, filename)

        return GenerateResponse(
            success=True,
            filename=filename,
            message="PDF generated successfully",
            tokens_used=len(ai_content.split())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/api/download/{filename}")
async def download_pdf(filename: str):
    """Download generated PDF"""
    filepath = f"output/{filename}"

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=filename
    )


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn

    print("Starting AI PDF Generator API...")
    print("Make sure Ollama is running with: ollama run llama2")
    uvicorn.run(app, host="0.0.0.0", port=8000)