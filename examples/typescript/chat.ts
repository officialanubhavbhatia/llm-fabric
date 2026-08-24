import { MyVista } from "myvista";

const client = new MyVista();
const response = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "Hello" }],
});
if ("text" in response) {
  console.log(response.text, response.requestId, response.fabric.servedModel);
}
