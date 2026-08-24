import { MyVista } from "myvista";

const client = new MyVista();
const decision = await client.intents.classify("debug this python traceback");
console.log(decision);
