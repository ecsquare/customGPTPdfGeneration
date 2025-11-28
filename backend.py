# AI PDF Generator MVP - Enhanced Version with Tables and Advanced Formatting
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib import colors
from datetime import datetime
import os
import re
import json

# Initialize FastAPI
app = FastAPI(title="AI PDF Generator", version="2.0.0")

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
4. Financial Summary (if applicable, format as TABLE with headers: Metric|Q1|Q2|Q3|Q4)
5. Recommendations (2-3 paragraphs)

For tables, use this format:
TABLE_START
Header1|Header2|Header3
Row1Col1|Row1Col2|Row1Col3
Row2Col1|Row2Col2|Row2Col3
TABLE_END

Use professional business language and provide specific, actionable insights. Do not use markdown formatting like ** or __.""",
        "style": "formal"
    },
    "invoice": {
        "name": "Invoice",
        "prompt_template": """Generate a professional invoice based on these requirements:

{user_prompt}

IMPORTANT: Structure your response EXACTLY as follows:

INVOICE_INFO_START
Invoice Number: [generate a number]
Invoice Date: [current date]
Due Date: [30 days from now]
INVOICE_INFO_END

COMPANY_INFO_START
From: [Company name and full address]
To: [Client name and full address]
COMPANY_INFO_END

ITEMS_TABLE_START
Item/Description|Quantity|Unit Price|Total
[Item 1 description]|[qty]|$[price]|$[total]
[Item 2 description]|[qty]|$[price]|$[total]
[Item 3 description]|[qty]|$[price]|$[total]
ITEMS_TABLE_END

TOTALS_START
Subtotal: $[amount]
Tax (10%): $[amount]
Total Due: $[amount]
TOTALS_END

PAYMENT_INFO_START
Payment Terms: [terms]
Payment Methods: [methods accepted]
Notes: [any additional notes]
PAYMENT_INFO_END

Generate realistic values. Use the pipe symbol | to separate columns in tables.""",
        "style": "structured"
    },
    "resume": {
        "name": "Resume/CV",
        "prompt_template": """Create a professional resume based on:

{user_prompt}

Structure with these sections:

HEADER_START
[Full Name]
[Email] | [Phone] | [LinkedIn] | [Location]
HEADER_END

Professional Summary:
[2-3 sentences highlighting key qualifications]

Work Experience:
[Company Name] - [Job Title]
[Start Date] - [End Date] | [Location]
• [Achievement with quantified result]
• [Achievement with quantified result]
• [Achievement with quantified result]

[Repeat for 2-3 positions]

SKILLS_TABLE_START
Category|Skills
Technical|[skill1, skill2, skill3, skill4]
Languages|[language1, language2, language3]
Tools|[tool1, tool2, tool3, tool4]
SKILLS_TABLE_END

Education:
[Degree] in [Field]
[University Name] | [Graduation Year] | GPA: [if notable]

Certifications:
• [Certification 1]
• [Certification 2]

Use action verbs and quantify achievements where possible. Do not use markdown formatting.""",
        "style": "professional"
    },
    "contract": {
        "name": "Contract",
        "prompt_template": """Draft a professional contract based on:

{user_prompt}

Structure the contract as follows:

CONTRACT_HEADER_START
[CONTRACT TYPE]
Agreement made on [DATE]
CONTRACT_HEADER_END

PARTIES_TABLE_START
Party|Details
Party A (Provider)|[Full name/company, address, contact]
Party B (Client)|[Full name/company, address, contact]
PARTIES_TABLE_END

1. EFFECTIVE DATE AND DURATION
[Start date, end date, duration details]

2. SCOPE OF WORK/SERVICES
• [Specific deliverable 1]
• [Specific deliverable 2]
• [Specific deliverable 3]

3. PAYMENT TERMS
PAYMENT_TABLE_START
Milestone|Amount|Due Date
[Milestone 1]|$[amount]|[date]
[Milestone 2]|$[amount]|[date]
[Final Payment]|$[amount]|[date]
PAYMENT_TABLE_END

Total Contract Value: $[total]

4. TERMS AND CONDITIONS
• [Term 1]
• [Term 2]
• [Term 3]

5. CONFIDENTIALITY
[Confidentiality clause details]

6. TERMINATION CONDITIONS
[Termination terms]

SIGNATURES_START
Party A: _________________________ Date: _________
Party B: _________________________ Date: _________
SIGNATURES_END

Use formal legal language appropriate for business contracts.""",
        "style": "legal"
    }
}


def clean_text_for_pdf(text):
    """Convert markdown-style formatting to ReportLab HTML tags"""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
    text = text.replace('&', '&amp;')
    text = re.sub(r'<(?![bi/>])', '&lt;', text)
    text = re.sub(r'(?<![bi/])>', '&gt;', text)
    return text


def create_table_from_data(data_lines, table_type="default"):
    """Create a formatted table from pipe-separated data"""

    # Parse table data
    rows = []
    for line in data_lines:
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')]
            rows.append(cells)

    if not rows:
        return None

    # Create table
    table = Table(rows, repeatRows=1)

    # Different styles for different table types
    if table_type == "invoice_items":
        style = TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

            # Body
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f9fafb')),
            ('TEXTCOLOR', (0, 1), (-1, -1), black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),  # Center quantity, price, total
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Left align description

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ])
    elif table_type == "header":
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#d1d5db')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])
    else:
        # Default table style
        style = TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

            # Alternating row colors
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffffff')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f9fafb')]),

            # Body
            ('TEXTCOLOR', (0, 1), (-1, -1), black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#d1d5db')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ])

    table.setStyle(style)
    return table


def parse_invoice_content(content, styles):
    """Special parser for invoice template with structured data"""

    elements = []

    # Title style
    title_style = ParagraphStyle(
        'InvoiceTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=HexColor('#1e3a8a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    # Section styles
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=HexColor('#1e3a8a'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    info_style = ParagraphStyle(
        'InfoText',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
    )

    total_style = ParagraphStyle(
        'TotalText',
        parent=styles['Normal'],
        fontSize=11,
        fontName='Helvetica-Bold',
        spaceAfter=4,
        alignment=TA_RIGHT,
    )

    # Extract sections using regex
    invoice_info = re.search(r'INVOICE_INFO_START(.*?)INVOICE_INFO_END', content, re.DOTALL)
    company_info = re.search(r'COMPANY_INFO_START(.*?)COMPANY_INFO_END', content, re.DOTALL)
    items_table = re.search(r'ITEMS_TABLE_START(.*?)ITEMS_TABLE_END', content, re.DOTALL)
    totals = re.search(r'TOTALS_START(.*?)TOTALS_END', content, re.DOTALL)
    payment_info = re.search(r'PAYMENT_INFO_START(.*?)PAYMENT_INFO_END', content, re.DOTALL)

    # Add INVOICE title
    elements.append(Paragraph("INVOICE", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Invoice basic info
    if invoice_info:
        info_lines = [line.strip() for line in invoice_info.group(1).strip().split('\n') if line.strip()]
        info_data = [[line.split(':', 1)[0].strip(), line.split(':', 1)[1].strip()] for line in info_lines if
                     ':' in line]
        if info_data:
            info_table = Table(info_data, colWidths=[2 * inch, 3 * inch])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(info_table)
            elements.append(Spacer(1, 0.2 * inch))

    # Company information
    if company_info:
        company_lines = [line.strip() for line in company_info.group(1).strip().split('\n') if line.strip()]

        # Split into From and To
        from_section = []
        to_section = []
        current_section = None

        for line in company_lines:
            if line.startswith('From:'):
                current_section = 'from'
                from_section.append(line.replace('From:', '').strip())
            elif line.startswith('To:'):
                current_section = 'to'
                to_section.append(line.replace('To:', '').strip())
            elif current_section == 'from':
                from_section.append(line)
            elif current_section == 'to':
                to_section.append(line)

        company_data = [
            ['FROM:', 'TO:'],
            ['\n'.join(from_section), '\n'.join(to_section)]
        ]

        company_table = Table(company_data, colWidths=[3 * inch, 3 * inch])
        company_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, 1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f3f4f6')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(company_table)
        elements.append(Spacer(1, 0.3 * inch))

    # Items table
    if items_table:
        table_lines = [line.strip() for line in items_table.group(1).strip().split('\n') if
                       line.strip() and '|' in line]
        items_tbl = create_table_from_data(table_lines, "invoice_items")
        if items_tbl:
            elements.append(items_tbl)
            elements.append(Spacer(1, 0.2 * inch))

    # Totals section
    if totals:
        total_lines = [line.strip() for line in totals.group(1).strip().split('\n') if line.strip()]
        for line in total_lines:
            if 'Total Due' in line:
                # Make total due prominent
                para = Paragraph(f'<b><font size="14">{line}</font></b>', total_style)
            else:
                para = Paragraph(line, total_style)
            elements.append(para)
        elements.append(Spacer(1, 0.3 * inch))

    # Payment information
    if payment_info:
        elements.append(Paragraph("Payment Information", section_style))
        payment_lines = [line.strip() for line in payment_info.group(1).strip().split('\n') if line.strip()]
        for line in payment_lines:
            para = Paragraph(clean_text_for_pdf(line), info_style)
            elements.append(para)

    return elements


def parse_content_to_elements(content, styles, template_name):
    """Parse AI-generated content into styled PDF elements"""

    # Special handling for invoice
    if template_name == "invoice" and "INVOICE_INFO_START" in content:
        return parse_invoice_content(content, styles)

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

        # Check for table markers
        if 'TABLE_START' in line or '_TABLE_START' in line:
            table_lines = []
            i += 1
            while i < len(lines) and 'TABLE_END' not in lines[i]:
                if lines[i].strip():
                    table_lines.append(lines[i].strip())
                i += 1

            if table_lines:
                table = create_table_from_data(table_lines)
                if table:
                    elements.append(table)
                    elements.append(Spacer(1, 0.2 * inch))
            i += 1
            continue

        # Detect section headings
        if re.match(r'^\d+\.\s+[A-Z]', line) or (line.isupper() and len(line) < 60):
            cleaned = clean_text_for_pdf(line)
            para = Paragraph(cleaned, heading_style)
            elements.append(para)

        # Detect sub-headings
        elif re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:?\s*$', line) and len(line) < 50:
            cleaned = clean_text_for_pdf(line)
            para = Paragraph(cleaned, subheading_style)
            elements.append(para)

        # Detect bullet points
        elif re.match(r'^[-*•]\s+', line) or re.match(r'^\d+\)\s+', line):
            cleaned = clean_text_for_pdf(line)
            cleaned = re.sub(r'^[-*•]\s+', '• ', cleaned)
            cleaned = re.sub(r'^\d+\)\s+', '• ', cleaned)
            para = Paragraph(cleaned, bullet_style)
            elements.append(para)

        # Regular paragraph
        else:
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

    # Only add title for non-invoice templates (invoice has its own title)
    if template_name != "invoice":
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

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

    # Parse and add content
    content_elements = parse_content_to_elements(content, styles, template_name)
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
        "version": "2.0.0",
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
        if request.template not in TEMPLATES:
            raise HTTPException(status_code=400, detail="Invalid template")

        template_config = TEMPLATES[request.template]
        prompt = PromptTemplate(
            input_variables=["user_prompt"],
            template=template_config["prompt_template"]
        )

        chain = prompt | llm | StrOutputParser()

        print(f"Generating content for: {request.prompt[:50]}...")
        ai_content = chain.invoke({"user_prompt": request.prompt})

        print("Generated content preview:")
        print(ai_content[:300])

        os.makedirs("output", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/{request.template}_{timestamp}.pdf"

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