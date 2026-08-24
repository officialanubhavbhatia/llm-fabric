import { MyVista } from "myvista";

const client = new MyVista();
const response = await client.responses.create({ input: "Debug this Python program" });
console.log(response.output_text);
