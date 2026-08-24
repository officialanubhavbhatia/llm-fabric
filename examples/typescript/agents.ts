import { MyVista, UnsupportedError } from "myvista";

const client = new MyVista();
try {
  await client.agents.run();
} catch (error) {
  if (error instanceof UnsupportedError) console.log(error.message);
}
