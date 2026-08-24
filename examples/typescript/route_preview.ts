import { MyVista } from "myvista";

const client = new MyVista();
const plan = await client.routes.preview({
  model: "auto",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(plan.explanation);
