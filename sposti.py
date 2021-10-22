from email.mime.text import MIMEText
import smtplib
def send_email(email,pdf):
    from_email = "kielioppisovellus@gmail.com"
    from_password = "iX8Wqq3o"
    to_email = email

    subject = "Tuloksesi"
    message = "Tiedostot liitteenä"

    msg=MIMEText(message,"html")
    msg['Subject']=subject
    msg['To']=to_email
    msg['From']=from_email

    gmail=smtplib.SMTP('smtp.gmail.com',587)
    gmail.ehlo()
    gmail.starttls()
    gmail.login(from_password,from_password)
    gmail.send_message(msg)