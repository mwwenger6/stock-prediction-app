interface User {
    email: string;
    password: string;
    id: number;
    createdAt: string;
    typeId: number;
    isVerified: boolean;
    verificationCode: string;
    typeName: string;
}
export default User;