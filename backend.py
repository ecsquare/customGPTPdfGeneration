# AI PDF Generator MVP - Enhanced Version with Better Styling
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
from reportlab.lib.colors import HexColor
from datetime import datetime
import os
import re

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

# Initialize Ollama
llm = OllamaLLM(model="llama3.1:8b", temperature=0)


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
3. Key Findings (bullet points format - use • prefix)
4. Recommendations (2-3 paragraphs)

Use professional business language and provide specific, actionable insights. Do not use markdown formatting like ** or __.""",
        "style": "formal"
    },
    "invoice": {
        "name": "Invoice",
        "prompt_template": """Generate a detailed invoice based on these requirements:

{user_prompt}

Include:
- Company and client information
- Invoice number and date
- Itemized list of services/products with descriptions (use • prefix for items)
- Individual prices and quantities
- Subtotal, tax (if applicable), and total amount
- Payment terms and methods

Format as a clear, professional invoice. Do not use markdown formatting.""",
        "style": "structured"
    },
    "resume": {
        "name": "Resume/CV",
        "prompt_template": """Create a professional resume based on:

{user_prompt}

Include sections for:
- Professional Summary (2-3 sentences)
- Work Experience (with bullet points using • prefix for achievements)
- Technical Skills
- Education
- Certifications (if applicable)

Use action verbs and quantify achievements where possible. Do not use markdown formatting.""",
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
- Terms and conditions (use numbered points)
- Confidentiality clause
- Termination conditions
- Signature section

Use formal legal language appropriate for business contracts. Do not use markdown formatting.""",
        "style": "legal"
    }
}


def clean_text_for_pdf(text):
    """Convert markdown-style formatting to ReportLab HTML tags"""

    # Remove markdown bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Remove markdown italic (*text* or _text_)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)

    # Escape XML special characters that aren't part of our tags
    text = text.replace('&', '&amp;')
    text = re.sub(r'<(?![bi/>])', '&lt;', text)
    text = re.sub(r'(?<![bi/])>', '&gt;', text)

    return text


def parse_content_to_elements(content, styles):
    """Parse AI-generated content into styled PDF elements"""

    elements = []
    lines = content.split('\n')

    # Custom styles
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#1e3a8a'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=HexColor('#2563eb'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14,
    )

    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=6,
        leftIndent=20,
        bulletIndent=10,
        leading=14,
    )

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            elements.append(Spacer(1, 0.1 * inch))
            i += 1
            continue

        # Detect section headings (numbered or all caps short lines)
        if re.match(r'^\d+\.\s+[A-Z]', line) or (line.isupper() and len(line) < 60):
            cleaned = clean_text_for_pdf(line)
            para = Paragraph(cleaned, heading_style)
            elements.append(para)

        # Detect sub-headings (capitalized words)
        elif re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:?\s*$', line) and len(line) < 50:
            cleaned = clean_text_for_pdf(line)
            para = Paragraph(cleaned, subheading_style)
            elements.append(para)

        # Detect bullet points (lines starting with -, *, •, or numbered)
        elif re.match(r'^[-*•]\s+', line) or re.match(r'^\d+\)\s+', line):
            cleaned = clean_text_for_pdf(line)
            # Remove the bullet/dash and add proper bullet
            cleaned = re.sub(r'^[-*•]\s+', '• ', cleaned)
            cleaned = re.sub(r'^\d+\)\s+', '• ', cleaned)
            para = Paragraph(cleaned, bullet_style)
            elements.append(para)

        # Regular paragraph
        else:
            # Collect multi-line paragraphs
            paragraph_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not re.match(r'^[-*•]\s+', lines[j].strip()) and not re.match(
                    r'^\d+\.\s+', lines[j].strip()):
                paragraph_lines.append(lines[j].strip())
                j += 1

            paragraph_text = ' '.join(paragraph_lines)
            cleaned = clean_text_for_pdf(paragraph_text)
            para = Paragraph(cleaned, body_style)
            elements.append(para)
            elements.append(Spacer(1, 0.05 * inch))

            i = j - 1

        i += 1

    return elements


def create_pdf(content: str, template_name: str, filename: str):
    """Generate PDF from AI-generated content with proper styling"""

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=50,
    )

    elements = []
    styles = getSampleStyleSheet()

    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1e3a8a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    # Add title
    template_title = TEMPLATES[template_name]["name"]
    title = Paragraph(template_title, title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2 * inch))

    # Add generation date
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#6b7280'),
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    date_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    date_para = Paragraph(date_text, date_style)
    elements.append(date_para)
    elements.append(Spacer(1, 0.3 * inch))

    # Parse and add content with proper styling
    content_elements = parse_content_to_elements(content, styles)
    elements.extend(content_elements)

    # Add footer
    elements.append(Spacer(1, 0.5 * inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=HexColor('#9ca3af'),
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

        # Create LangChain prompt
        prompt = PromptTemplate(
            input_variables=["user_prompt"],
            template=template_config["prompt_template"]
        )

        # Create chain using LCEL
        chain = prompt | llm | StrOutputParser()

        # Generate content
        print(f"Generating content for: {request.prompt[:50]}...")
        ai_content = chain.invoke({"user_prompt": request.prompt})

        print("Generated content preview:")
        print(ai_content[:200])

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
        print(f"Error: {str(e)}")
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


if __name__ == "__main__":
    import uvicorn

    print("Starting AI PDF Generator API...")
    print("Make sure Ollama is running with: ollama run llama3.1:8b")
    uvicorn.run(app, host="0.0.0.0", port=8000)