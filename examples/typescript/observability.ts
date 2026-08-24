import { MyVista } from "myvista";

const client = new MyVista();
await client.chat.completions.create({
  model: "auto",
  messages: [{ role: "user", content: "Hello" }],
});
const listed = await client.traces.list();
console.log(listed);
