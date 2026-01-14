import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# 配置
ids = [0, 1, 2, 3, 4]
marker_size = 5 * cm
margin = 2 * cm
c = canvas.Canvas("ArUco_5cm_Print.pdf", pagesize=A4)
y_position = 29.7 * cm - marker_size - margin

for mid in ids:
    # 生成高精度码
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(dictionary, mid, 1000)
    cv2.imwrite(f"{mid}.png", img)
    
    # 绘制到PDF (居中)
    x_position = (21.0 * cm - marker_size) / 2
    c.drawImage(f"{mid}.png", x_position, y_position, width=marker_size, height=marker_size)
    c.setFont("Helvetica", 8)
    c.drawString(x_position, y_position - 0.4*cm, f"ID: {mid} | Size: 5cm")
    
    # 移动到下一个位置 (5cm码 + 2cm白边 + 1cm间距)
    y_position -= (marker_size + margin + 1*cm)
    if y_position < margin:
        c.showPage()
        y_position = 29.7 * cm - marker_size - margin

c.save()