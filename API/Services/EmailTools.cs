using System;
using System.IO;
using System.Net;
using System.Net.Mail;
using Microsoft.Extensions.Configuration;

namespace Stock_Prediction_API.Services
{
    public class EmailTools
    {
        private readonly IConfiguration _config;
        private readonly string senderEmail, senderPassword, emailVerificationFilePath;
        private readonly SmtpClient client;

        public EmailTools(IConfiguration config, IWebHostEnvironment env)
        {
            _config = config;
            senderEmail = _config.GetValue<string>("GoogleEmail:Email");
            senderPassword = _config.GetValue<string>("GoogleEmail:Password");
            emailVerificationFilePath = Path.Combine(env.WebRootPath, "Emails", "EmailVerification.html");

            client = new SmtpClient
            {
                UseDefaultCredentials = false,
                Credentials = new NetworkCredential(senderEmail, senderPassword),
                Host = "smtp.gmail.com",
                EnableSsl = true,
                Port = 587
            };
        }

        public void SendVerificationEmail(string toEmail, string code)
        {
            string emailBody = File.ReadAllText(emailVerificationFilePath);
            string verificationLink = "https://stockgenie.net/" + code;
            emailBody = emailBody.Replace("{viewingLink}", verificationLink);

            MailMessage mailMessage = new()
            {
                From = new MailAddress(senderEmail),
                Subject = "Verify Your StockGenie Account",
                Body = emailBody,
                IsBodyHtml = true,
            };
            mailMessage.To.Add(toEmail);

            try
            {
                client.Send(mailMessage);
                Console.WriteLine("Verification email sent successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to send verification email. Exception: {ex.Message}");
            }
        }
    }
}
