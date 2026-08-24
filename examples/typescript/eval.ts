import { MyVista } from "myvista";

const client = new MyVista();
const run = await client.evals.run("ci");
console.log(run.suite_name, run.metrics);
