using System;
using System.Net;
using System.IO;
using System.Net.Mail;
using Microsoft.Extensions.Configuration;

namespace Stock_Prediction_API.Services
{
    public class EmailTools
    {
        private readonly IConfiguration _config;
        private readonly string senderEmail, senderPassword, emailVerificationFilePath;
        private readonly SmtpClient client;

        public EmailTools(IConfiguration config)
        {
            _config = config;
            senderEmail = _config.GetValue<string>("GoogleEmail:Email");
            senderPassword = _config.GetValue<string>("GoogleEmail:Password");
            emailVerificationFilePath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "Emails", "EmailVerification.html");

            // Update SMTP settings for Gmail
            client = new SmtpClient
            {
                UseDefaultCredentials = false,
                Credentials = new NetworkCredential(senderEmail, senderPassword),
                Host = "smtp.gmail.com", // Gmail SMTP server
                EnableSsl = true,
                Port = 587 // Use 587 for TLS
            };
        }

        public void SendVerificationEmail(string toEmail, string code)
        {
            string emailBody = File.ReadAllText(emailVerificationFilePath);
            string verificationLink = "https://stockgenie.net/EmailVerification/" + code;
            emailBody = emailBody.Replace("{viewingLink}", verificationLink);

            MailMessage mailMessage = new()
            {
                From = new MailAddress(senderEmail),
                Subject = "Verify Your StockGenie Account",
                Body = emailBody,
                IsBodyHtml = true,
            };
            mailMessage.To.Add(toEmail);
            client.Send(mailMessage);
        }
    }
}
