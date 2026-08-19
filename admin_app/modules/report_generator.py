from pathlib import Path
from xml.sax.saxutils import escape
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def generate_report(patient_id,row,prediction,recommendations,output_dir="reports"):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); report_file=output/f"{patient_id}_adherence_report.pdf"
    styles=getSampleStyleSheet(); doc=SimpleDocTemplate(str(report_file),pagesize=A4)
    data=[["Metric","Value"],["Patient ID",escape(str(patient_id))],["Report date",str(date.today())],["Completed adherence history",str(int(row.get("history_days",0)))+" days"],["Average adherence",f"{float(row['mean_adherence']):.1%}"],["Missed doses",str(int(row['missed_doses']))],["Late doses",str(int(row['late_doses']))],["Predicted non-adherence risk",f"{prediction['risk_probability']:.1%}"],["Risk classification",escape(prediction['risk_label'])]]
    story=[Paragraph("Medication Adherence Report",styles["Title"]),Spacer(1,12),Paragraph("MediTrack research/decision-support summary",styles["Heading2"]),Spacer(1,8),Table(data,repeatRows=1,style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dff6f2")),("GRID",(0,0),(-1,-1),0.5,colors.grey),("PADDING",(0,0),(-1,-1),6)])),Spacer(1,15),Paragraph("Recommended Actions",styles["Heading2"])]
    story += [Paragraph("• "+escape(str(r)),styles["Normal"]) for r in recommendations]
    story += [Spacer(1,15),Paragraph("Model methodology: GradientBoosting classifier trained on bundled synthetic data using a 24-day historical window to predict whether future adherence is at least 70%. Live predictions use only data available before the prediction cutoff.",styles["Normal"]),Spacer(1,8),Paragraph("This is educational decision-support output. It is not clinical validation, a diagnosis, or a replacement for professional medical advice.",styles["Italic"])]
    doc.build(story); return str(report_file)
