using System;
using System.IO;
using System.Net;
using System.Net.Mail;
using Microsoft.Extensions.Configuration;
using Stock_Prediction_API.ViewModel;

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
            string verificationLink = "https://stockgenie.net/Verification/" + code;
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

        public void SendStockEmail(StockEmailViewModel model)
        {
            string emailBody = File.ReadAllText(emailVerificationFilePath);
            string stockLink = "https://stockgenie.net/Stock/" + model.Ticker;
            emailBody = emailBody.Replace("{viewingLink}", stockLink)
                .Replace("{stockName}", model.StockName)
                .Replace("{ticker}", model.Ticker)
                .Replace("{indication}", model.Indication);

            MailMessage mailMessage = new()
            {
                From = new MailAddress(senderEmail),
                Subject = $"{model.Ticker} IS ON THE MOVE",
                Body = emailBody,
                IsBodyHtml = true,
            };
            mailMessage.To.Add(model.Email);
            client.Send(mailMessage);
        }
    }
}
