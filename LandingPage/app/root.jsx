import {
  Links,
  LiveReload,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "@remix-run/react";
import styles from "~/styles/main.css"
import t_style from "~/tailwind/tailwind.css"
// import Sidebar from "./Components/sidebar";
// import NavBar from "./Components/navbar";

export default function App() {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {/* <header>
         <NavBar></NavBar>
        </header> */}
        <Outlet />
        <ScrollRestoration />
        <Scripts />
        <LiveReload />
      </body>
    </html>
  );
}
export function links()
{
  return ([{ rel: "stylesheet", href: styles }, { rel: "stylesheet", href: t_style }]);
}

