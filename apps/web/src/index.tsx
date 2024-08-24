/* @refresh reload */
import "./index.css";
import { render } from "solid-js/web";
import { Route, Router } from "@solidjs/router";

const Home = () => {
  return <div class="text-red-500">Home</div>;
};

const NotFoundRedirect = () => {
  window.location.href = "/";

  return <></>;
};

const root = document.getElementById("root");

render(
  () => (
    <Router>
      <Route path="/" component={Home} />
      <Route path="/:not_found" component={NotFoundRedirect} />
    </Router>
  ),

  root!,
);
