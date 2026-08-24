import { MyVista } from "myvista";

const client = new MyVista();
const response = await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: 'Reply with JSON only: {"ok": true}' }],
  response_format: { type: "json_object" },
});
if ("text" in response) console.log(response.text);
