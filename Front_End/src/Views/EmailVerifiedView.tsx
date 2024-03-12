import React, { useEffect } from "react";
import config from "../config";
interface VerifyUserProps {
  code: string;
}

const VerifyUser: React.FC<VerifyUserProps> = ({ code }) => {
  useEffect(() => {
    const verifyUser = async () => {
      try {
        // const response = await fetch(config.verifyUser(code), {
            const response = await fetch("https://localhost:7212/Home/VerifyUser/" + code, {
          method: "POST",
        });

        if (response.ok) {
          const data = await response.json();
          console.log("Verification response:", data);
        } else {
          console.error("Verification failed:", response.statusText);
        }
      } catch (error) {
        console.error("Verification error:", error);
      }
    };

    verifyUser();
  }, [code]);

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1 style={{ color: "green", fontSize: "3em" }}>
        Your Account Has Been Verified
      </h1>
    </div>
  );
};

export default VerifyUser;
